import os
import json
import uuid
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
from qdrant_client import QdrantClient, models as qdrant_models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
from tqdm import tqdm

# Step 1: Configuration
METADATA_DIR = 
IMAGE_DIR = 
QDRANT_URL = 
QDRANT_API_KEY = 
QDRANT_COLLECTION_NAME = "fashion_visual_recommender"
VECTOR_SIZE = 512
NAMESPACE_UUID = uuid.NAMESPACE_DNS

# Step 2: Custom ResNet Feature Extractor Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for feature extraction: {device}")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class CustomResNet(nn.Module):
    def __init__(self, block, layers, vector_size=VECTOR_SIZE):
        super(CustomResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, vector_size)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

custom_resnet_model = CustomResNet(ResidualBlock, [2, 2, 2, 2], vector_size=VECTOR_SIZE).to(device)
custom_resnet_model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_custom_resnet_embedding(path: str):
    try:
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            features = custom_resnet_model(image)
            features /= features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy()
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        return None

# Step 3: Qdrant Setup
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

def setup_qdrant_collection():
    try:
        print(f"Attempting to create collection '{QDRANT_COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print("Collection created successfully.")
        print("Creating payload indexes...")
        filterable_fields = [
            "clothing_type", "gender_suitability", "occasion_suitability",
            "sleeve_type", "neckline", "closure_type", "dominant_color",
            "secondary_color", "pattern_type", "original_id"
        ]
        for field in filterable_fields:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name=field,
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
                wait=True
            )
        print("Payload indexes created.")
    except UnexpectedResponse as e:
        if "already exists" in str(e):
            print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists. Skipping creation.")
        else:
            raise e

def populate_knowledge_base():
    print("Starting knowledge base population with Custom (Untrained) ResNet embeddings...")
    metadata_files = [f for f in os.listdir(METADATA_DIR) if f.endswith('.json')]
    points_to_upsert = []
    for metadata_file in tqdm(metadata_files, desc="Processing Files"):
        image_id_str = os.path.splitext(metadata_file)[0]
        metadata_path = os.path.join(METADATA_DIR, metadata_file)
        image_path = os.path.join(IMAGE_DIR, f"{image_id_str}.jpg")
        if not os.path.exists(image_path): continue
        vector = get_custom_resnet_embedding(image_path)
        if vector is None: continue
        with open(metadata_path, 'r') as f: metadata = json.load(f)
        point_id = str(uuid.uuid5(NAMESPACE_UUID, image_id_str))
        metadata['original_id'] = image_id_str
        points_to_upsert.append(qdrant_models.PointStruct(id=point_id, vector=vector.flatten().tolist(), payload=metadata))
        if len(points_to_upsert) >= 50:
            client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points_to_upsert, wait=True)
            points_to_upsert = []
    if points_to_upsert:
        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points_to_upsert, wait=True)
    print("Knowledge base population complete!")
    print(f"Total points in collection: {client.get_collection(QDRANT_COLLECTION_NAME).points_count}")

# Step 4: Search and Recommendation Functions
def recommend_visually(query_id_str: str, filters: dict = None, limit: int = 5):
    print(f"\n--- Running Visual Recommendation for: {query_id_str} ---")
    query_point_response, _ = client.scroll(
        collection_name=QDRANT_COLLECTION_NAME,
        scroll_filter=qdrant_models.Filter(must=[qdrant_models.FieldCondition(key="original_id", match=qdrant_models.MatchValue(value=query_id_str))]),
        limit=1, with_vectors=True
    )
    if not query_point_response:
        print(f"Error: Could not find item with original_id: {query_id_str}")
        return []
    query_vector = query_point_response[0].vector
    query_point_id = query_point_response[0].id
    filter_conditions = []
    if filters:
        for key, value in filters.items():
            filter_conditions.append(qdrant_models.FieldCondition(key=key, match=qdrant_models.MatchValue(value=value)))
    search_results = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_vector,
        query_filter=qdrant_models.Filter(must=filter_conditions) if filter_conditions else None,
        limit=limit + 1,
    ).points
    return [result for result in search_results if result.id != query_point_id][:limit]

# Step 5: Visualization Functions
def visualize_recommendations(query_id_str, results):
    if not results:
        print("No results to visualize.")
        return
    num_results = len(results)
    fig, axes = plt.subplots(1, num_results + 1, figsize=(5 * (num_results + 1), 5))
    fig.suptitle("Recommendations from Untrained Custom ResNet (Results are Random)", fontsize=16)
    query_image_path = os.path.join(IMAGE_DIR, f"{query_id_str}.jpg")
    try:
        query_img = Image.open(query_image_path)
        axes[0].imshow(query_img)
        axes[0].set_title(f"Query Item:\n{query_id_str}")
        axes[0].axis('off')
    except FileNotFoundError:
        axes[0].set_title(f"Query Image Not Found:\n{query_id_str}")
        axes[0].axis('off')
    for i, res in enumerate(results):
        rec_id, rec_score = res.payload.get('original_id', 'N/A'), res.score
        rec_image_path = os.path.join(IMAGE_DIR, f"{rec_id}.jpg")
        ax = axes[i+1]
        try:
            rec_img = Image.open(rec_image_path)
            ax.imshow(rec_img)
            ax.set_title(f"Rec #{i+1}: {rec_id}\nScore: {rec_score:.4f}")
        except FileNotFoundError:
            ax.set_title(f"Image Not Found:\n{rec_id}")
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def visualize_clusters(sample_limit=500):
    print(f"\n--- Visualizing clusters for a sample of {sample_limit} items ---")
    try:
        points, _ = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=sample_limit,
            with_payload=True,
            with_vectors=True
        )
        if not points:
            print("No points found in the collection to visualize.")
            return
        vectors = np.array([point.vector for point in points])
        labels = [point.payload.get('clothing_type', 'Unknown') for point in points]
        n_pca_components = min(50, len(points) - 1, vectors.shape[1])
        print(f"Running PCA to reduce dimensions to {n_pca_components}...")
        pca = PCA(n_components=n_pca_components)
        vectors_pca = pca.fit_transform(vectors)
        print("Running t-SNE for 2D visualization...")
        tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        vectors_2d = tsne_2d.fit_transform(vectors_pca)
        plt.figure(figsize=(16, 12))
        unique_labels = sorted(list(set(labels)))
        palette = sns.color_palette("bright", len(unique_labels))
        sns.scatterplot(
            x=vectors_2d[:, 0], y=vectors_2d[:, 1], hue=labels,
            palette=palette, legend="full", s=50, alpha=0.7
        )
        plt.title('2D Cluster Visualization (Untrained Custom ResNet)', fontsize=16)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(title='Clothing Type', bbox_to_anchor=(1.05, 1), loc=2)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()
    except Exception as e:
        print(f"An error occurred during cluster visualization: {e}")

# Step 6: Main Execution Block
if __name__ == "__main__":
    # setup_qdrant_collection()
    # populate_knowledge_base()
    visualize_clusters(sample_limit=1500)
    print("\n" + "="*50)
    print("PERFORMANCE TEST WITH CUSTOM (UNTRAINED) RESNET")
    print("="*50)
    query_image_id = "00006_00"
    visual_results = recommend_visually(query_id_str=query_image_id, limit=4)
    print(f"\n[RESULTS] Found {len(visual_results)} items for {query_image_id}:")
    print("(Note: These results are based on random vectors and are not meaningful)")
    for result in visual_results:
        rec_id = result.payload.get('original_id', 'N/A')
        rec_type = result.payload.get('clothing_type', 'Unknown')
        print(f"  - Recommended: {rec_id} (Type: {rec_type}, Score: {result.score:.4f})")
    visualize_recommendations(query_image_id, visual_results)
