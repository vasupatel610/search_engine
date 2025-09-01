# 🛍️ Fashion Retail Search Engine

A powerful semantic search engine for fashion products that returns the top 3 closest recommendations using advanced AI techniques.

## ✨ Features

- **Semantic Search**: Uses sentence transformers for intelligent product matching
- **Top 3 Results**: Always returns the 3 most relevant products
- **Hybrid Search**: Combines semantic and lexical matching for better results
- **Faceted Filtering**: Automatically applies filters based on query context
- **RESTful API**: Clean API endpoints for easy integration
- **Web Interface**: Beautiful Streamlit-based UI for testing
- **Real-time Recommendations**: Get similar product suggestions instantly

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Search Engine Server

```bash
python scripts/search.py
```

The server will start on `http://localhost:8000`

### 3. Test the API

```bash
python scripts/test_search_api.py
```

### 4. Use the Web Interface

```bash
streamlit run scripts/search_web_interface.py
```

## 🔌 API Endpoints

### Search Products
```
GET /search?query={search_query}&top_k={number_of_results}
```

**Parameters:**
- `query` (required): Your search query
- `top_k` (optional): Number of results (1-10, default: 3)
- `hybrid_boost` (optional): Lexical boost factor (default: 0.2)
- `filter_facets` (optional): Apply facet filters (default: true)

**Example:**
```bash
curl "http://localhost:8000/search?query=party%20heels&top_k=3"
```

### Get Recommendations
```
GET /recommend/{product_id}?top_k={number_of_recommendations}
```

**Parameters:**
- `product_id` (required): ID of the product
- `top_k` (optional): Number of recommendations (1-10, default: 3)

**Example:**
```bash
curl "http://localhost:8000/recommend/PROD001?top_k=3"
```

### Other Endpoints
- `GET /` - API information
- `GET /categories` - List all product categories
- `GET /stats` - Dataset statistics
- `GET /health` - Health check

## 📊 Example Queries

Try these example searches:

- **"party heels"** - Find elegant party footwear
- **"work laptop bag"** - Professional work accessories
- **"kids winter jackets"** - Children's winter clothing
- **"navy running shoes"** - Athletic footwear in specific color
- **"cotton t-shirt"** - Comfortable casual wear
- **"leather handbag"** - Premium leather accessories

## 🧠 How It Works

### 1. **Semantic Understanding**
- Uses the `all-MiniLM-L6-v2` model for text embeddings
- Converts product descriptions and search queries to vector representations
- Finds semantic similarity between queries and products

### 2. **Hybrid Scoring**
- **Semantic Score**: Cosine similarity between query and product embeddings
- **Lexical Score**: Token overlap between query and product text
- **Business Score**: Stock status and price optimization
- **Final Score**: Weighted combination of all scores

### 3. **Smart Filtering**
- Automatically detects product categories from queries
- Applies occasion-based filtering (party, work, casual, etc.)
- Considers age groups and materials
- Filters by color and product type

### 4. **Top 3 Selection**
- Ranks all products by final score
- Returns exactly 3 most relevant results
- Ensures diverse and high-quality recommendations

## 🛠️ Technical Details

### Model Architecture
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Dimension**: 384
- **Normalization**: L2 normalization for cosine similarity
- **Batch Processing**: 64 samples per batch

### Performance Features
- **Cached Embeddings**: Pre-computed product embeddings stored in `embeddings.npy`
- **FastAPI**: High-performance async web framework
- **Efficient Search**: O(n) complexity for similarity search
- **Response Time**: Typically <100ms for search queries

### Data Processing
- **Text Normalization**: Handles synonyms, accents, and variations
- **Facet Inference**: Automatically extracts product attributes from queries
- **Material Validation**: Ensures materials match product categories
- **Price Optimization**: Slight boost for products near median price

## 📁 File Structure

```
fashion_retail_analytics/
├── scripts/
│   ├── search.py                 # Main search engine server
│   ├── test_search_api.py        # API testing script
│   └── search_web_interface.py   # Streamlit web interface
├── data/
│   └── raw/
│       └── fashion_products_100_clean.csv  # Product dataset
├── embeddings.npy                # Cached product embeddings
├── requirements.txt               # Python dependencies
└── SEARCH_ENGINE_README.md       # This file
```

## 🔧 Configuration

### Environment Variables
- `CSV_PATH`: Path to your product dataset
- `EMB_PATH`: Path to store embeddings cache
- `MODEL_NAME`: Sentence transformer model to use

### Model Parameters
- `TOP_K_DEFAULT`: Default number of results (set to 3)
- `HYBRID_BOOST`: Lexical score weight (default: 0.2)
- `BATCH_SIZE`: Embedding computation batch size (default: 64)

## 🚀 Deployment

### Local Development
```bash
# Terminal 1: Start search engine
python scripts/search.py

# Terminal 2: Start web interface
streamlit run scripts/search_web_interface.py

# Terminal 3: Test API
python scripts/test_search_api.py
```

### Production Deployment
```bash
# Using uvicorn directly
uvicorn scripts.search:app --host 0.0.0.0 --port 8000 --workers 4

# Using gunicorn
gunicorn scripts.search:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 📈 Performance Tips

1. **Use Cached Embeddings**: The system automatically caches embeddings in `embeddings.npy`
2. **Optimize Queries**: Be specific in your search terms for better results
3. **Batch Requests**: For multiple searches, consider batching API calls
4. **Monitor Response Times**: Use the `/health` endpoint to check performance

## 🐛 Troubleshooting

### Common Issues

**Server won't start:**
- Check if port 8000 is available
- Ensure all dependencies are installed
- Verify the CSV file path is correct

**No search results:**
- Check if the dataset is loaded correctly
- Verify the embeddings are computed
- Try simpler search queries

**Slow performance:**
- Check if embeddings are cached
- Monitor system resources
- Consider using a smaller model

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python scripts/search.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Sentence Transformers**: For semantic text understanding
- **FastAPI**: For high-performance API framework
- **Streamlit**: For beautiful web interfaces
- **Pandas & NumPy**: For efficient data processing

---

**Happy Searching! 🎉**

For questions or support, please open an issue in the repository.
