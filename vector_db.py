import os
import json
from qdrant_client import QdrantClient, models as qdrant_models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
from tqdm import tqdm
import numpy as np
import uuid
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# --- 1. CONFIGURATION ---
METADATA_DIR = "C:/CodeVSCSummers/AIMS_Vton_stuff/features_extracted"
IMAGE_DIR = "C:/CodeVSCSummers/AIMS_Vton_stuff/clothes_tryon_dataset/test/cloth"

# --- QDRANT CLOUD CONFIGURATION ---
QDRANT_URL = "https://c977d41b-092f-4746-8055-e0c1974ed673.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.-IpipKy3ymQr1qZIAvAzNZ87K1e0Qxd8B2MNKTWY07w"

QDRANT_COLLECTION_NAME = "fashion_visual_recommender"
VECTOR_SIZE = 2048

# A consistent namespace for generating UUIDs from filenames
NAMESPACE_UUID = uuid.NAMESPACE_DNS

# --- 2. IMAGE FEATURE EXTRACTOR SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for feature extraction: {device}")

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_vector(image_path):
    """Loads an image, processes it, and returns its feature vector."""
    try:
        img = Image.open(image_path).convert("RGB")
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0).to(device)
        with torch.no_grad():
            features = model(batch_t)
            vector = features.squeeze().cpu().numpy()
        return vector
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# --- 3. QDRANT SETUP ---
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

def setup_qdrant_collection():
    """Creates the collection and payload indexes if they don't exist."""
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
        # This is the expected error if the collection already exists.
        if "already exists" in str(e):
            print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists. Skipping creation.")
        else:
            # If it's another error, we should raise it.
            raise e


def populate_knowledge_base():
    """Populates the database with image vectors and metadata."""
    print("Starting knowledge base population...")
    metadata_files = [f for f in os.listdir(METADATA_DIR) if f.endswith('.json')]
    points_to_upsert = []

    for metadata_file in tqdm(metadata_files, desc="Processing Files"):
        image_id_str = os.path.splitext(metadata_file)[0]
        metadata_path = os.path.join(METADATA_DIR, metadata_file)
        image_path = os.path.join(IMAGE_DIR, f"{image_id_str}.jpg")

        if not os.path.exists(image_path): continue
        vector = get_image_vector(image_path)
        if vector is None: continue

        with open(metadata_path, 'r') as f: metadata = json.load(f)

        point_id = str(uuid.uuid5(NAMESPACE_UUID, image_id_str))
        metadata['original_id'] = image_id_str
        points_to_upsert.append(qdrant_models.PointStruct(id=point_id, vector=vector.tolist(), payload=metadata))
        
        if len(points_to_upsert) >= 50:
            client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points_to_upsert, wait=True)
            points_to_upsert = []

    if points_to_upsert:
        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points_to_upsert, wait=True)

    print("Knowledge base population complete!")
    print(f"Total points in collection: {client.get_collection(QDRANT_COLLECTION_NAME).points_count}")

# --- 4. SEARCH AND RECOMMENDATION FUNCTIONS ---

def recommend_visually(query_id_str: str, filters: dict = None, limit: int = 5):
    """Finds visually similar items, with optional metadata filtering (hybrid search)."""
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

    search_results = client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=qdrant_models.Filter(must=filter_conditions) if filter_conditions else None,
        limit=limit + 1,
    )
    return [result for result in search_results if result.id != query_point_id][:limit]

# --- 5. VISUALIZATION FUNCTIONS ---

def visualize_recommendations(query_id_str, results):
    """Creates a plot showing the query image and its top recommended images."""
    if not results:
        print("No results to visualize.")
        return
    num_results = len(results)
    fig, axes = plt.subplots(1, num_results + 1, figsize=(5 * (num_results + 1), 5))
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
    plt.tight_layout()
    plt.show()

def visualize_clusters(sample_limit=500):
    """
    Fetches a sample of vectors, runs t-SNE, and plots the resulting 2D clusters.
    NOTE: This requires scikit-learn: pip install scikit-learn
    """
    print(f"\n--- Visualizing clusters for a sample of {sample_limit} items ---")
    try:
        # Fetch a sample of points from Qdrant
        points, _ = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=sample_limit,
            with_payload=True,
            with_vectors=True
        )
        if not points:
            print("No points found in the collection to visualize.")
            return

        # Extract vectors and metadata for coloring
        vectors = np.array([point.vector for point in points])
        labels = [point.payload.get('clothing_type', 'Unknown') for point in points]
        
        print("Running t-SNE... (this may take a moment)")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        vectors_2d = tsne.fit_transform(vectors)
        
        # Create the plot
        plt.figure(figsize=(16, 12))
        
        # Define a custom color palette
        unique_labels = sorted(list(set(labels)))
        # Using a bright, distinct color palette from seaborn
        palette = sns.color_palette("bright", len(unique_labels))
        color_map = dict(zip(unique_labels, palette))

        sns.scatterplot(
            x=vectors_2d[:, 0],
            y=vectors_2d[:, 1],
            hue=labels,
            palette=color_map, # Use the custom distinct color map
            legend="full",
            s=50, # Marker size
            alpha=0.7 # Marker transparency
        )
        plt.title('2D Cluster Visualization of Image Embeddings (t-SNE)', fontsize=16)
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend(title='Clothing Type', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend
        plt.show()

    except ImportError:
        print("\nERROR: scikit-learn is not installed. Please run 'pip install scikit-learn' to use this feature.")
    except Exception as e:
        print(f"An error occurred during cluster visualization: {e}")

# --- 6. HOW TO USE ---
if __name__ == "__main__":
    setup_qdrant_collection()
    
    # Step 1: Populate the database. Run this once.
    # populate_knowledge_base()

    # --- Example 1: Visualize the Clusters ---
    # This will generate a scatter plot showing how your items are grouped.
    visualize_clusters(sample_limit=1000) # Increased sample size for better visualization

    # --- Example 2: Visual Recommendation and Performance Test ---
    print("\n" + "="*50)
    print("VISUAL PERFORMANCE TEST")
    print("="*50)
    
    query_image_id = "00069_00" # An item to find recommendations for
    visual_results = recommend_visually(query_id_str=query_image_id, limit=4)
    
    print(f"\n[RESULTS] Found {len(visual_results)} visually similar items for {query_image_id}:")
    for result in visual_results:
        rec_id = result.payload.get('original_id', 'N/A')
        rec_type = result.payload.get('clothing_type', 'Unknown')
        print(f"  - Recommended: {rec_id} (Type: {rec_type}, Score: {result.score:.4f})")
    
    # Visualize the recommendation results in a plot
    visualize_recommendations(query_image_id, visual_results)

