# ##############################################################################
# # Enhanced E-commerce Search with Single Letter Brand/Category Suggestions - FIXED
# import streamlit as st
# import streamlit.components.v1 as components
# import pandas as pd
# import threading
# from fastapi import FastAPI, Query
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import nest_asyncio
# from rapidfuzz import fuzz, process
# import json
# import os
# import warnings
# from collections import defaultdict, Counter
# import re
# import socket
# import time

# warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

# # Apply nest_asyncio for running uvicorn inside Streamlit
# nest_asyncio.apply()

# # Initialize FastAPI app
# api_app = FastAPI()
# api_app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global variables for enhanced search
# product_data = {}
# brand_index = {}  # Single letter -> brands
# category_index = {}  # Single letter -> categories
# brand_products = defaultdict(list)
# category_products = defaultdict(list)
# search_index = {}
# popular_brands = []
# popular_categories = []

# def find_free_port():
#     """Find a free port for the API server"""
#     for port in range(8001, 8020):
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             try:
#                 s.bind(('127.0.0.1', port))
#                 return port
#             except OSError:
#                 continue
#     return 8001  # fallback

# API_PORT = find_free_port()

# def clean_text(text):
#     """Clean and normalize text for better matching"""
#     if pd.isna(text) or text == '':
#         return ''
#     text = str(text).lower().strip()
#     text = re.sub(r'[^\w\s]', ' ', text)
#     text = ' '.join(text.split())
#     return text

# def extract_category_from_name(product_name):
#     """Extract category from product name using comprehensive keyword matching"""
#     product_name_lower = product_name.lower()
    
#     category_keywords = {
#         'shirts': ['shirt', 'shirts', 'formal shirt', 'casual shirt'],
#         'jeans': ['jeans', 'denim', 'jean'],
#         'dresses': ['dress', 'dresses', 'gown', 'frock'],
#         'kurtas': ['kurta', 'kurti', 'kurtis', 'ethnic wear'],
#         'sarees': ['saree', 'sari', 'sarees'],
#         'shoes': ['shoes', 'shoe', 'sneakers', 'boots', 'sandals', 'heels', 'loafers', 'footwear'],
#         't-shirts': ['t-shirt', 'tshirt', 't shirt', 'tee', 'round neck'],
#         'trousers': ['trousers', 'pants', 'chinos', 'formal pants'],
#         'shorts': ['shorts', 'short', 'bermuda'],
#         'tops': ['top', 'tops', 'blouse', 'tank top'],
#         'jackets': ['jacket', 'blazer', 'coat', 'hoodie', 'sweatshirt'],
#         'bags': ['bag', 'handbag', 'backpack', 'purse', 'sling bag', 'tote'],
#         'watches': ['watch', 'watches', 'timepiece'],
#         'sunglasses': ['sunglasses', 'glasses', 'shades', 'eyewear'],
#         'belts': ['belt', 'belts', 'leather belt'],
#         'caps': ['cap', 'hat', 'caps', 'beanie'],
#         'wallets': ['wallet', 'wallets', 'purse'],
#         'perfumes': ['perfume', 'fragrance', 'cologne', 'deo', 'deodorant'],
#         'jewelry': ['jewelry', 'jewellery', 'necklace', 'earrings', 'bracelet', 'ring'],
#         'makeup': ['lipstick', 'foundation', 'mascara', 'eyeliner', 'concealer', 'makeup'],
#         'innerwear': ['bra', 'panty', 'brief', 'boxer', 'innerwear', 'lingerie'],
#         'sportswear': ['track', 'sports', 'gym', 'fitness', 'athletic'],
#         'ethnic': ['lehenga', 'salwar', 'dupatta', 'ethnic', 'traditional']
#     }
    
#     for category, keywords in category_keywords.items():
#         for keyword in keywords:
#             if keyword in product_name_lower:
#                 return category
    
#     return 'others'

# def build_single_letter_indexes():
#     """Build indexes for single letter brand and category suggestions"""
#     global brand_index, category_index, popular_brands, popular_categories
    
#     # Build brand index (first letter -> brands) - filter out meaningless brands
#     brand_count = Counter()
#     excluded_brands = {'unknown', 'brand', 'nan', '', 'unbranded', 'generic'}
    
#     for brand, products in brand_products.items():
#         if (brand and len(brand) > 1 and 
#             brand.lower() not in excluded_brands and 
#             not brand.lower().startswith('set ') and
#             not brand.lower().startswith('pack ') and
#             len(products) >= 5):  # Only brands with at least 5 products
            
#             first_letter = brand[0].lower()
#             if first_letter.isalpha():  # Only alphabetic letters
#                 if first_letter not in brand_index:
#                     brand_index[first_letter] = []
#                 brand_index[first_letter].append({
#                     'name': brand,
#                     'count': len(products),
#                     'display_name': brand.title()
#                 })
#                 brand_count[brand] = len(products)
    
#     # Sort brands by popularity within each letter
#     for letter in brand_index:
#         brand_index[letter].sort(key=lambda x: x['count'], reverse=True)
#         brand_index[letter] = brand_index[letter][:6]  # Top 6 per letter
    
#     # Build category index (first letter -> categories) - ensure meaningful categories
#     category_count = Counter()
#     valid_categories = {'shirts', 'shoes', 'jeans', 'dresses', 'kurtas', 'sarees', 
#                        't-shirts', 'trousers', 'shorts', 'tops', 'jackets', 'bags', 
#                        'watches', 'sunglasses', 'belts', 'caps', 'wallets', 'perfumes', 
#                        'jewelry', 'makeup', 'innerwear', 'sportswear', 'ethnic'}
    
#     for category, products in category_products.items():
#         if (category and len(category) > 1 and 
#             category.lower() in valid_categories and 
#             len(products) >= 10):  # Only categories with at least 10 products
            
#             first_letter = category[0].lower()
#             if first_letter.isalpha():  # Only alphabetic letters
#                 if first_letter not in category_index:
#                     category_index[first_letter] = []
#                 category_index[first_letter].append({
#                     'name': category,
#                     'count': len(products),
#                     'display_name': category.title()
#                 })
#                 category_count[category] = len(products)
    
#     # Sort categories by popularity within each letter
#     for letter in category_index:
#         category_index[letter].sort(key=lambda x: x['count'], reverse=True)
#         category_index[letter] = category_index[letter][:5]  # Top 5 per letter
    
#     # Store popular brands and categories for quick access
#     popular_brands = [item[0] for item in brand_count.most_common(15)]
#     popular_categories = [item[0] for item in category_count.most_common(10)]

# def load_product_data():
#     """Load and process product data with enhanced indexing"""
#     global product_data, brand_products, category_products, search_index
    
#     csv_path = "/home/artisans15/projects/fashion_retail_analytics/data/raw/myntra_products_catalog.csv"
    
#     if not os.path.exists(csv_path):
#         st.error(f"CSV file not found at: {csv_path}")
#         return False
    
#     try:
#         df = pd.read_csv(csv_path)
#         required_columns = ['product_name', 'ProductBrand', 'Gender', 'price']
#         missing_columns = [col for col in required_columns if col not in df.columns]
#         if missing_columns:
#             st.error(f"Missing columns in CSV: {missing_columns}")
#             return False
        
#         # Process more products for better brand/category coverage
#         df = df.head(15000)
#         st.info(f"Processing {len(df)} products for enhanced search...")
        
#         processed_count = 0
        
#         for idx, row in df.iterrows():
#             try:
#                 product_name = str(row['product_name']).strip()
#                 brand = str(row['ProductBrand']).strip()
#                 gender = str(row['Gender']).strip()
#                 price = row['price']
                
#                 if not product_name or product_name.lower() == 'nan':
#                     continue
                
#                 clean_name = clean_text(product_name)
#                 category = extract_category_from_name(product_name)
                
#                 if not brand or brand.lower() == 'nan':
#                     brand = 'Unknown'
#                 if not gender or gender.lower() == 'nan':
#                     gender = 'Unisex'
                
#                 try:
#                     price = float(price) if price and str(price).lower() != 'nan' else 0
#                 except:
#                     price = 0
                
#                 color = ''
#                 if 'PrimaryColor' in df.columns:
#                     color = str(row.get('PrimaryColor', '')).strip()
                
#                 # Store product data
#                 product_key = clean_name
#                 product_data[product_key] = {
#                     'name': product_name,
#                     'brand': brand,
#                     'gender': gender,
#                     'price': price,
#                     'category': category,
#                     'color': color,
#                     'original_name': product_name
#                 }
                
#                 # Enhanced indexing for brands and categories with better filtering
#                 clean_brand = clean_text(brand)
#                 if (clean_brand and len(clean_brand) > 1 and 
#                     not clean_brand.startswith('set ') and 
#                     not clean_brand.startswith('pack ') and
#                     clean_brand not in ['unknown', 'brand', 'nan', 'unbranded']):
#                     brand_products[clean_brand].append(product_key)
                
#                 if category and len(category) > 0:
#                     category_products[category].append(product_key)
                
#                 # Build comprehensive search index
#                 search_index[clean_name] = product_key
                
#                 # Index individual words
#                 words = clean_name.split()
#                 for word in words:
#                     if len(word) > 1:  # Include 2+ letter words
#                         if word not in search_index:
#                             search_index[word] = []
#                         if isinstance(search_index[word], list):
#                             search_index[word].append(product_key)
#                         else:
#                             search_index[word] = [search_index[word], product_key]
                
#                 # Index brand
#                 if clean_brand and len(clean_brand) > 1:
#                     if clean_brand not in search_index:
#                         search_index[clean_brand] = []
#                     if isinstance(search_index[clean_brand], list):
#                         search_index[clean_brand].append(product_key)
#                     else:
#                         search_index[clean_brand] = [search_index[clean_brand], product_key]
                
#                 # Index category
#                 if category and len(category) > 1:
#                     if category not in search_index:
#                         search_index[category] = []
#                     if isinstance(search_index[category], list):
#                         search_index[category].append(product_key)
#                     else:
#                         search_index[category] = [search_index[category], product_key]
                
#                 processed_count += 1
                
#             except Exception as e:
#                 continue
        
#         # Build single letter indexes after processing all products
#         build_single_letter_indexes()
        
#         st.success(f"Loaded {processed_count} products successfully!")
#         st.info(f"Indexed {len(search_index)} search terms, {len(brand_products)} brands, {len(category_products)} categories")
#         return True
        
#     except Exception as e:
#         st.error(f"Error loading CSV: {str(e)}")
#         return False

# def get_smart_suggestions(query: str, limit: int = 10):
#     """Enhanced suggestions with single letter brand/category support"""
#     if not query:
#         return []
    
#     original_query = query
#     query_clean = clean_text(query)
#     suggestions = []
#     seen = set()
    
#     # Special handling for single character queries - ONLY brands and categories
#     if len(original_query) == 1:
#         letter = original_query.lower()
        
#         # Add brand suggestions for this letter (higher priority)
#         if letter in brand_index:
#             for brand_info in brand_index[letter][:5]:  # Top 5 brands
#                 suggestions.append({
#                     'suggestion': brand_info['display_name'],
#                     'type': 'Brand',
#                     'scope': f"{brand_info['count']} products",
#                     'category': 'brand',
#                     'gender': 'All',
#                     'price': 0,
#                     '_score': 100 + brand_info['count'] * 0.1
#                 })
        
#         # Add category suggestions for this letter
#         if letter in category_index:
#             for cat_info in category_index[letter][:5]:  # Top 5 categories
#                 suggestions.append({
#                     'suggestion': cat_info['display_name'],
#                     'type': 'Category',
#                     'scope': f"{cat_info['count']} items",
#                     'category': cat_info['name'],
#                     'gender': 'All',
#                     'price': 0,
#                     '_score': 95 + cat_info['count'] * 0.1
#                 })
        
#         # Sort by score and return ONLY brands and categories for single letter
#         suggestions.sort(key=lambda x: x['_score'], reverse=True)
#         return suggestions[:limit]
    
#     # For multi-character queries, use existing logic with enhancements
    
#     # 1. Exact matches
#     if query_clean in search_index:
#         matches = search_index[query_clean]
#         if isinstance(matches, str):
#             matches = [matches]
        
#         for match in matches[:3]:
#             if match in product_data and match not in seen:
#                 product = product_data[match]
#                 suggestions.append({
#                     'suggestion': product['original_name'],
#                     'type': 'Exact Match',
#                     'scope': f"{product['brand']} • {product['gender']}",
#                     'category': product['category'],
#                     'gender': product['gender'],
#                     'price': product['price'],
#                     '_score': 100
#                 })
#                 seen.add(match)
    
#     # 2. Brand matching (enhanced)
#     for brand in brand_products.keys():
#         if query_clean in brand or brand.startswith(query_clean):
#             brand_display = brand.title()
#             if brand_display not in [s['suggestion'] for s in suggestions]:
#                 suggestions.append({
#                     'suggestion': brand_display,
#                     'type': 'Brand',
#                     'scope': f"{len(brand_products[brand])} products",
#                     'category': 'brand',
#                     'gender': 'All',
#                     'price': 0,
#                     '_score': 95
#                 })
    
#     # 3. Category matching (enhanced)
#     for category in category_products.keys():
#         if query_clean in category or category.startswith(query_clean):
#             category_display = category.title()
#             if category_display not in [s['suggestion'] for s in suggestions]:
#                 suggestions.append({
#                     'suggestion': category_display,
#                     'type': 'Category',
#                     'scope': f"{len(category_products[category])} items",
#                     'category': category,
#                     'gender': 'All',
#                     'price': 0,
#                     '_score': 92
#                 })
    
#     # 4. Prefix matches
#     for key, matches in search_index.items():
#         if key.startswith(query_clean) and len(suggestions) < limit * 2:
#             if isinstance(matches, str):
#                 matches = [matches]
            
#             for match in matches[:2]:
#                 if match in product_data and match not in seen:
#                     product = product_data[match]
#                     suggestions.append({
#                         'suggestion': product['original_name'],
#                         'type': 'Autocomplete',
#                         'scope': f"{product['brand']} • {product['gender']}",
#                         'category': product['category'],
#                         'gender': product['gender'],
#                         'price': product['price'],
#                         '_score': 88
#                     })
#                     seen.add(match)
    
#     # 5. Fuzzy matching (for 2+ characters)
#     if len(query_clean) >= 2 and len(suggestions) < limit:
#         all_product_names = [(data['original_name'], key) for key, data in product_data.items()]
#         product_names_only = [name for name, _ in all_product_names]
        
#         fuzzy_matches = process.extract(
#             original_query, 
#             product_names_only, 
#             scorer=fuzz.partial_ratio, 
#             limit=limit
#         )
        
#         for match_name, score, _ in fuzzy_matches:
#             if score > 65:
#                 matching_key = None
#                 for orig_name, key in all_product_names:
#                     if orig_name == match_name:
#                         matching_key = key
#                         break
                
#                 if matching_key and matching_key not in seen:
#                     product = product_data[matching_key]
#                     suggestions.append({
#                         'suggestion': product['original_name'],
#                         'type': 'Similar',
#                         'scope': f"{product['brand']} • {product['gender']}",
#                         'category': product['category'],
#                         'gender': product['gender'],
#                         'price': product['price'],
#                         '_score': score * 0.8
#                     })
#                     seen.add(matching_key)
                    
#                     if len(suggestions) >= limit:
#                         break
    
#     # Sort by score and return
#     suggestions.sort(key=lambda x: x['_score'], reverse=True)
#     return suggestions[:limit]

# @api_app.get("/suggestions")
# async def suggestions_endpoint(q: str = Query(..., min_length=1), limit: int = 10):
#     """API endpoint for enhanced search suggestions"""
#     return get_smart_suggestions(q, limit)

# @api_app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "products_loaded": len(product_data),
#         "brands_indexed": len(brand_products),
#         "categories_indexed": len(category_products),
#         "search_terms": len(search_index),
#         "port": API_PORT
#     }

# def run_api():
#     try:
#         uvicorn.run(api_app, host="127.0.0.1", port=API_PORT, log_level="error")
#     except Exception as e:
#         st.error(f"API server error: {e}")

# def initialize_app():
#     """Initialize the enhanced autocomplete system"""
#     if 'app_initialized' not in st.session_state:
#         with st.spinner("Loading enhanced product search..."):
#             success = load_product_data()
#             if success:
#                 st.session_state.app_initialized = True
#                 st.session_state.api_port = API_PORT
#                 api_thread = threading.Thread(target=run_api, daemon=True)
#                 api_thread.start()
#                 st.session_state.api_thread_started = True
#                 # Give API time to start
#                 time.sleep(2)
#                 return True
#             else:
#                 return False
#     return True

# # Streamlit UI
# st.title("Enhanced Product Search")
# st.markdown("*Type even a single letter to see brands and categories!*")

# if initialize_app():
#     # Show enhanced statistics
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric("Products", len(product_data))
    
#     with col2:
#         st.metric("Brands", len(brand_products))
    
#     with col3:
#         st.metric("Categories", len(category_products))
    
#     with col4:
#         st.metric("API Port", st.session_state.get('api_port', API_PORT))
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import threading
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import nest_asyncio
from rapidfuzz import fuzz, process
import json
import os
import warnings
from collections import defaultdict, Counter
import re
import socket
import time

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

# Apply nest_asyncio for running uvicorn inside Streamlit
nest_asyncio.apply()

# Initialize FastAPI app
api_app = FastAPI()
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NEW: Gender Relevance Dictionary ---
GENDER_RELEVANCE = {
    'saree': {'Women'},
    'kurta': {'Men', 'Women'},
    'lehenga': {'Women'},
    'shirt': {'Men', 'Women'},
    'dress': {'Women'},
    'suit': {'Men', 'Women'},
    'jeans': {'Men', 'Women'},
    'shoes': {'Men', 'Women'},
    'sandals': {'Men', 'Women'},
    'heels': {'Women'},
    'watch': {'Men', 'Women'},
    'handbag': {'Women'},
    'wallet': {'Men', 'Women'},
    't-shirt': {'Men', 'Women'},
    'trousers': {'Men', 'Women'},
    'shorts': {'Men', 'Women'},
    'skirt': {'Women'},
    'blouse': {'Women'},
    'jackets': {'Men', 'Women'},
    'coats': {'Men', 'Women'},
    'sneakers': {'Men', 'Women'},
    'flipflops': {'Men', 'Women'},
    'cap': {'Men', 'Women'},
    'hat': {'Men', 'Women'},
    'belt': {'Men', 'Women'},
    'scarf': {'Men', 'Women'},
    'gloves': {'Men', 'Women'},
    'socks': {'Men', 'Women'},
    'tie': {'Men'},
    'cufflinks': {'Men'},
    'lingerie': {'Women'},
    'nightwear': {'Men', 'Women'},
    'sportswear': {'Men', 'Women'},
    'swimwear': {'Men', 'Women'},
    'backpack': {'Men', 'Women'},
    'jewelry': {'Women'},
    'earrings': {'Women'},
    'necklace': {'Women'},
    'bracelet': {'Women'},
    'sunglasses': {'Men', 'Women'},
    'perfume': {'Men', 'Women'},
    'formal shoes': {'Men'},
    'casual shoes': {'Men', 'Women'},
    'loafers': {'Men', 'Women'},
    'tops': {'Women'},
    'bags': {'Men', 'Women'},
    'innerwear': {'Men', 'Women'},
    'makeup': {'Women'},
    'ethnic': {'Men', 'Women'}
}
# A few common categories from your `extract_category_from_name` function were added above.

# Global variables for enhanced search
product_data = {}
brand_index = {}  # Single letter -> brands
category_index = {}  # Single letter -> categories
brand_products = defaultdict(list)
category_products = defaultdict(list)
search_index = {}
popular_brands = []
popular_categories = []

def find_free_port():
    """Find a free port for the API server"""
    for port in range(8001, 8020):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    return 8001  # fallback

API_PORT = find_free_port()

def clean_text(text):
    """Clean and normalize text for better matching"""
    if pd.isna(text) or text == '':
        return ''
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text

def extract_category_from_name(product_name):
    """Extract category from product name using comprehensive keyword matching"""
    product_name_lower = product_name.lower()
    
    category_keywords = {
        'shirts': ['shirt', 'shirts', 'formal shirt', 'casual shirt'],
        'jeans': ['jeans', 'denim', 'jean'],
        'dresses': ['dress', 'dresses', 'gown', 'frock'],
        'kurtas': ['kurta', 'kurti', 'kurtis', 'ethnic wear'],
        'sarees': ['saree', 'sari', 'sarees'],
        'shoes': ['shoes', 'shoe', 'sneakers', 'boots', 'sandals', 'heels', 'loafers', 'footwear'],
        't-shirts': ['t-shirt', 'tshirt', 't shirt', 'tee', 'round neck'],
        'trousers': ['trousers', 'pants', 'chinos', 'formal pants'],
        'shorts': ['shorts', 'short', 'bermuda'],
        'tops': ['top', 'tops', 'blouse', 'tank top'],
        'jackets': ['jacket', 'blazer', 'coat', 'hoodie', 'sweatshirt'],
        'bags': ['bag', 'handbag', 'backpack', 'purse', 'sling bag', 'tote'],
        'watches': ['watch', 'watches', 'timepiece'],
        'sunglasses': ['sunglasses', 'glasses', 'shades', 'eyewear'],
        'belts': ['belt', 'belts', 'leather belt'],
        'caps': ['cap', 'hat', 'caps', 'beanie'],
        'wallets': ['wallet', 'wallets', 'purse'],
        'perfumes': ['perfume', 'fragrance', 'cologne', 'deo', 'deodorant'],
        'jewelry': ['jewelry', 'jewellery', 'necklace', 'earrings', 'bracelet', 'ring'],
        'makeup': ['lipstick', 'foundation', 'mascara', 'eyeliner', 'concealer', 'makeup'],
        'innerwear': ['bra', 'panty', 'brief', 'boxer', 'innerwear', 'lingerie'],
        'sportswear': ['track', 'sports', 'gym', 'fitness', 'athletic'],
        'ethnic': ['lehenga', 'salwar', 'dupatta', 'ethnic', 'traditional']
    }
    
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in product_name_lower:
                return category
    
    return 'others'

def build_single_letter_indexes():
    """Build indexes for single letter brand and category suggestions"""
    global brand_index, category_index, popular_brands, popular_categories
    
    # Build brand index (first letter -> brands) - filter out meaningless brands
    brand_count = Counter()
    excluded_brands = {'unknown', 'brand', 'nan', '', 'unbranded', 'generic'}
    
    for brand, products in brand_products.items():
        if (brand and len(brand) > 1 and 
            brand.lower() not in excluded_brands and 
            not brand.lower().startswith('set ') and
            not brand.lower().startswith('pack ') and
            len(products) >= 5):  # Only brands with at least 5 products
            
            first_letter = brand[0].lower()
            if first_letter.isalpha():  # Only alphabetic letters
                if first_letter not in brand_index:
                    brand_index[first_letter] = []
                brand_index[first_letter].append({
                    'name': brand,
                    'count': len(products),
                    'display_name': brand.title()
                })
                brand_count[brand] = len(products)
    
    # Sort brands by popularity within each letter
    for letter in brand_index:
        brand_index[letter].sort(key=lambda x: x['count'], reverse=True)
        brand_index[letter] = brand_index[letter][:6]  # Top 6 per letter
    
    # Build category index (first letter -> categories) - ensure meaningful categories
    category_count = Counter()
    valid_categories = {'shirts', 'shoes', 'jeans', 'dresses', 'kurtas', 'sarees', 
                       't-shirts', 'trousers', 'shorts', 'tops', 'jackets', 'bags', 
                       'watches', 'sunglasses', 'belts', 'caps', 'wallets', 'perfumes', 
                       'jewelry', 'makeup', 'innerwear', 'sportswear', 'ethnic'}
    
    for category, products in category_products.items():
        if (category and len(category) > 1 and 
            category.lower() in valid_categories and 
            len(products) >= 10):  # Only categories with at least 10 products
            
            first_letter = category[0].lower()
            if first_letter.isalpha():  # Only alphabetic letters
                if first_letter not in category_index:
                    category_index[first_letter] = []
                category_index[first_letter].append({
                    'name': category,
                    'count': len(products),
                    'display_name': category.title()
                })
                category_count[category] = len(products)
    
    # Sort categories by popularity within each letter
    for letter in category_index:
        category_index[letter].sort(key=lambda x: x['count'], reverse=True)
        category_index[letter] = category_index[letter][:5]  # Top 5 per letter
    
    # Store popular brands and categories for quick access
    popular_brands = [item[0] for item in brand_count.most_common(15)]
    popular_categories = [item[0] for item in category_count.most_common(10)]

def load_product_data():
    """Load and process product data with enhanced indexing"""
    global product_data, brand_products, category_products, search_index
    
    csv_path = "/home/artisans15/projects/fashion_retail_analytics/data/raw/myntra_products_catalog.csv"
    
    if not os.path.exists(csv_path):
        st.error(f"CSV file not found at: {csv_path}")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['product_name', 'ProductBrand', 'Gender', 'price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing columns in CSV: {missing_columns}")
            return False
        
        # Process more products for better brand/category coverage
        df = df.head(15000)
        st.info(f"Processing {len(df)} products for enhanced search...")
        
        processed_count = 0
        
        for idx, row in df.iterrows():
            try:
                product_name = str(row['product_name']).strip()
                brand = str(row['ProductBrand']).strip()
                gender = str(row['Gender']).strip()
                price = row['price']
                
                if not product_name or product_name.lower() == 'nan':
                    continue
                
                clean_name = clean_text(product_name)
                category = extract_category_from_name(product_name)
                
                if not brand or brand.lower() == 'nan':
                    brand = 'Unknown'
                if not gender or gender.lower() == 'nan':
                    gender = 'Unisex'
                
                try:
                    price = float(price) if price and str(price).lower() != 'nan' else 0
                except:
                    price = 0
                
                color = ''
                if 'PrimaryColor' in df.columns:
                    color = str(row.get('PrimaryColor', '')).strip()
                
                # Store product data
                product_key = clean_name
                product_data[product_key] = {
                    'name': product_name,
                    'brand': brand,
                    'gender': gender,
                    'price': price,
                    'category': category,
                    'color': color,
                    'original_name': product_name
                }
                
                # Enhanced indexing for brands and categories with better filtering
                clean_brand = clean_text(brand)
                if (clean_brand and len(clean_brand) > 1 and 
                    not clean_brand.startswith('set ') and 
                    not clean_brand.startswith('pack ') and
                    clean_brand not in ['unknown', 'brand', 'nan', 'unbranded']):
                    brand_products[clean_brand].append(product_key)
                
                if category and len(category) > 0:
                    category_products[category].append(product_key)
                
                # Build comprehensive search index
                search_index[clean_name] = product_key
                
                # Index individual words
                words = clean_name.split()
                for word in words:
                    if len(word) > 1:  # Include 2+ letter words
                        if word not in search_index:
                            search_index[word] = []
                        if isinstance(search_index[word], list):
                            search_index[word].append(product_key)
                        else:
                            search_index[word] = [search_index[word], product_key]
                
                # Index brand
                if clean_brand and len(clean_brand) > 1:
                    if clean_brand not in search_index:
                        search_index[clean_brand] = []
                    if isinstance(search_index[clean_brand], list):
                        search_index[clean_brand].append(product_key)
                    else:
                        search_index[clean_brand] = [search_index[clean_brand], product_key]
                
                # Index category
                if category and len(category) > 1:
                    if category not in search_index:
                        search_index[category] = []
                    if isinstance(search_index[category], list):
                        search_index[category].append(product_key)
                    else:
                        search_index[category] = [search_index[category], product_key]
                
                processed_count += 1
                
            except Exception as e:
                continue
        
        # Build single letter indexes after processing all products
        build_single_letter_indexes()
        
        st.success(f"Loaded {processed_count} products successfully!")
        st.info(f"Indexed {len(search_index)} search terms, {len(brand_products)} brands, {len(category_products)} categories")
        return True
        
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return False

def get_smart_suggestions(query: str, limit: int = 10):
    """Enhanced suggestions with single letter brand/category support"""
    if not query:
        return []
    
    original_query = query
    query_clean = clean_text(query)
    suggestions = []
    seen = set()
    
    # Special handling for single character queries - ONLY brands and categories
    if len(original_query) == 1:
        letter = original_query.lower()
        
        # Add brand suggestions for this letter (higher priority)
        if letter in brand_index:
            for brand_info in brand_index[letter][:5]:  # Top 5 brands
                suggestions.append({
                    'suggestion': brand_info['display_name'],
                    'type': 'Brand',
                    'scope': f"{brand_info['count']} products",
                    'category': 'brand',
                    'gender': 'Unisex',
                    'price': 0,
                    '_score': 100 + brand_info['count'] * 0.1
                })
        
        # Add category suggestions for this letter
        if letter in category_index:
            for cat_info in category_index[letter][:5]:  # Top 5 categories
                
                # --- MODIFIED: Add Gender Relevance ---
                category_name = cat_info['name']
                genders = GENDER_RELEVANCE.get(category_name, {'Men', 'Women'})
                if genders == {'Men', 'Women'}:
                    gender_display = 'For Men & Women'
                    gender_value = 'Unisex'
                elif genders == {'Women'}:
                    gender_display = 'For Women'
                    gender_value = 'Women'
                else:
                    gender_display = 'For Men'
                    gender_value = 'Men'
                
                suggestions.append({
                    'suggestion': cat_info['display_name'],
                    'type': 'Category',
                    'scope': f"{gender_display} • {cat_info['count']} items",
                    'category': category_name,
                    'gender': gender_value,
                    'price': 0,
                    '_score': 95 + cat_info['count'] * 0.1
                })
        
        # Sort by score and return ONLY brands and categories for single letter
        suggestions.sort(key=lambda x: x['_score'], reverse=True)
        return suggestions[:limit]
    
    # For multi-character queries, use existing logic with enhancements
    
    # 1. Exact matches
    if query_clean in search_index:
        matches = search_index[query_clean]
        if isinstance(matches, str):
            matches = [matches]
        
        for match in matches[:3]:
            if match in product_data and match not in seen:
                product = product_data[match]
                suggestions.append({
                    'suggestion': product['original_name'],
                    'type': 'Exact Match',
                    'scope': f"{product['brand']} • {product['gender']}",
                    'category': product['category'],
                    'gender': product['gender'],
                    'price': product['price'],
                    '_score': 100
                })
                seen.add(match)
    
    # 2. Brand matching (enhanced)
    for brand in brand_products.keys():
        if query_clean in brand or brand.startswith(query_clean):
            brand_display = brand.title()
            if brand_display not in [s['suggestion'] for s in suggestions]:
                suggestions.append({
                    'suggestion': brand_display,
                    'type': 'Brand',
                    'scope': f"{len(brand_products[brand])} products",
                    'category': 'brand',
                    'gender': 'Unisex',
                    'price': 0,
                    '_score': 95
                })
    
    # 3. Category matching (enhanced)
    for category in category_products.keys():
        if query_clean in category or category.startswith(query_clean):
            category_display = category.title()
            if category_display not in [s['suggestion'] for s in suggestions]:
                
                # --- MODIFIED: Add Gender Relevance ---
                genders = GENDER_RELEVANCE.get(category, {'Men', 'Women'})
                if genders == {'Men', 'Women'}:
                    gender_display = 'For Men & Women'
                    gender_value = 'Unisex'
                elif genders == {'Women'}:
                    gender_display = 'For Women'
                    gender_value = 'Women'
                else:
                    gender_display = 'For Men'
                    gender_value = 'Men'

                suggestions.append({
                    'suggestion': category_display,
                    'type': 'Category',
                    'scope': f"{gender_display} • {len(category_products[category])} items",
                    'category': category,
                    'gender': gender_value,
                    'price': 0,
                    '_score': 92
                })
    
    # 4. Prefix matches
    for key, matches in search_index.items():
        if key.startswith(query_clean) and len(suggestions) < limit * 2:
            if isinstance(matches, str):
                matches = [matches]
            
            for match in matches[:2]:
                if match in product_data and match not in seen:
                    product = product_data[match]
                    suggestions.append({
                        'suggestion': product['original_name'],
                        'type': 'Autocomplete',
                        'scope': f"{product['brand']} • {product['gender']}",
                        'category': product['category'],
                        'gender': product['gender'],
                        'price': product['price'],
                        '_score': 88
                    })
                    seen.add(match)
    
    # 5. Fuzzy matching (for 2+ characters)
    if len(query_clean) >= 2 and len(suggestions) < limit:
        all_product_names = [(data['original_name'], key) for key, data in product_data.items()]
        product_names_only = [name for name, _ in all_product_names]
        
        fuzzy_matches = process.extract(
            original_query, 
            product_names_only, 
            scorer=fuzz.partial_ratio, 
            limit=limit
        )
        
        for match_name, score, _ in fuzzy_matches:
            if score > 65:
                matching_key = None
                for orig_name, key in all_product_names:
                    if orig_name == match_name:
                        matching_key = key
                        break
                
                if matching_key and matching_key not in seen:
                    product = product_data[matching_key]
                    suggestions.append({
                        'suggestion': product['original_name'],
                        'type': 'Similar',
                        'scope': f"{product['brand']} • {product['gender']}",
                        'category': product['category'],
                        'gender': product['gender'],
                        'price': product['price'],
                        '_score': score * 0.8
                    })
                    seen.add(matching_key)
                    
                    if len(suggestions) >= limit:
                        break
    
    # Sort by score and remove duplicates before returning
    final_suggestions = []
    seen_suggestions = set()
    for s in sorted(suggestions, key=lambda x: x['_score'], reverse=True):
        if s['suggestion'] not in seen_suggestions:
            final_suggestions.append(s)
            seen_suggestions.add(s['suggestion'])
            
    return final_suggestions[:limit]

@api_app.get("/suggestions")
async def suggestions_endpoint(q: str = Query(..., min_length=1), limit: int = 10):
    """API endpoint for enhanced search suggestions"""
    return get_smart_suggestions(q, limit)

@api_app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "products_loaded": len(product_data),
        "brands_indexed": len(brand_products),
        "categories_indexed": len(category_products),
        "search_terms": len(search_index),
        "port": API_PORT
    }

def run_api():
    try:
        uvicorn.run(api_app, host="127.0.0.1", port=API_PORT, log_level="error")
    except Exception as e:
        st.error(f"API server error: {e}")

def initialize_app():
    """Initialize the enhanced autocomplete system"""
    if 'app_initialized' not in st.session_state:
        with st.spinner("Loading enhanced product search..."):
            success = load_product_data()
            if success:
                st.session_state.app_initialized = True
                st.session_state.api_port = API_PORT
                api_thread = threading.Thread(target=run_api, daemon=True)
                api_thread.start()
                st.session_state.api_thread_started = True
                # Give API time to start
                time.sleep(2)
                return True
            else:
                return False
    return True

# Streamlit UI
st.title("Enhanced Product Search")
st.markdown("*Type a product (e.g., 'blue shirt'), a brand, a category, or just a single letter to see relevant suggestions!*")

if initialize_app():
    # Show enhanced statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Products", len(product_data))
    
    with col2:
        st.metric("Brands", len(brand_products))
    
    with col3:
        st.metric("Categories", len(category_products))
    
    with col4:
        st.metric("API Port", st.session_state.get('api_port', API_PORT))
    # Enhanced search interface with faster response
    search_html = f"""
    <div style="position: relative; display: inline-block; width: 100%;">
        <input type="text" id="searchInput" placeholder="Try typing just 'n' for Nike, 's' for shirts, or any product name..." 
               style="width:100%; max-width:700px; font-size:18px; padding:16px; border: 3px solid #4CAF50; 
                      border-radius: 12px; outline: none; box-shadow: 0 3px 10px rgba(76,175,80,0.3);
                      transition: all 0.2s ease;"/>
        <div id="suggestions-container" style="
            border: 1px solid #ddd;
            max-height: 450px; 
            width: 100%;
            max-width: 700px;
            z-index: 1000; 
            background: white; 
            position: absolute;
            top: 100%;
            left: 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            border-radius: 0 0 12px 12px;
            display: none;
            overflow-y: auto;
        "></div>
        <div id="loading" style="display: none; padding: 10px; text-align: center; color: #666; font-size: 14px;">
            ⚡ Searching instantly...
        </div>
        <div id="error" style="display: none; padding: 10px; text-align: center; color: #f44336; font-size: 14px;">
            ❌ Search service unavailable
        </div>
    </div>
    
    <style>
    #searchInput:focus {{
        border-color: #2196F3;
        box-shadow: 0 3px 15px rgba(33,150,243,0.4);
        transform: translateY(-1px);
    }}
    
    .suggestion-item {{
        padding: 14px 16px;
        cursor: pointer;
        border-bottom: 1px solid #f0f0f0;
        transition: all 0.15s ease;
        font-size: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: relative;
    }}
    
    .suggestion-item:hover, .suggestion-item.selected {{
        background: linear-gradient(90deg, #f8f9ff 0%, #e8f4fd 100%);
        border-left: 4px solid #2196F3;
        padding-left: 12px;
    }}
    
    .suggestion-main {{
        display: flex;
        flex-direction: column;
        flex: 1;
    }}
    
    .suggestion-text {{
        color: #111;
        font-weight: 600;
        margin-bottom: 2px;
    }}
    
    .suggestion-scope {{
        font-size: 13px;
        color: #666;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    
    .suggestion-meta {{
        text-align: right;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 2px;
    }}
    
    .suggestion-type {{
        font-size: 12px;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .type-brand {{ background: #fff3e0; color: #f57c00; }}
    .type-category {{ background: #e8f5e8; color: #388e3c; }}
    .type-product {{ background: #f3e5f5; color: #7b1fa2; }}
    .type-exactmatch {{ background: #ffebee; color: #c62828; }}
    .type-autocomplete {{ background: #e3f2fd; color: #1976d2; }}
    .type-similar {{ background: #fce4ec; color: #c2185b; }}
    
    .price {{
        color: #ff5722;
        font-weight: 700;
        font-size: 14px;
    }}
    
    .single-letter-hint {{
        position: absolute;
        right: 10px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 11px;
        color: #999;
        background: #f5f5f5;
        padding: 2px 6px;
        border-radius: 8px;
    }}
    </style>
    
    <script>
    const searchInput = document.getElementById("searchInput");
    const suggestionsContainer = document.getElementById("suggestions-container");
    const loadingDiv = document.getElementById("loading");
    const errorDiv = document.getElementById("error");
    let searchTimeout;
    let selectedIndex = -1;
    const API_PORT = {st.session_state.get('api_port', API_PORT)};

    searchInput.addEventListener("input", function() {{
        clearTimeout(searchTimeout);
        let query = this.value;
        
        if (query.length === 0) {{
            hideSuggestions();
            return;
        }}
        
        showLoading();
        
        // Faster response for single characters
        let delay = query.length === 1 ? 150 : 200;
        searchTimeout = setTimeout(() => fetchSuggestions(query), delay);
    }});

    function showLoading() {{
        loadingDiv.style.display = 'block';
        errorDiv.style.display = 'none';
    }}

    function hideLoading() {{
        loadingDiv.style.display = 'none';
    }}

    function showError() {{
        errorDiv.style.display = 'block';
        loadingDiv.style.display = 'none';
    }}

    function fetchSuggestions(query) {{
        const url = `http://127.0.0.1:${{API_PORT}}/suggestions?q=${{encodeURIComponent(query)}}&limit=10`;
        
        fetch(url)
            .then(response => {{
                if (!response.ok) {{
                    throw new Error(`HTTP ${{response.status}}`);
                }}
                return response.json();
            }})
            .then(data => {{
                hideLoading();
                suggestionsContainer.innerHTML = "";
                
                if (data.length > 0) {{
                    data.forEach((item, index) => {{
                        let div = document.createElement('div');
                        div.className = 'suggestion-item';
                        
                        let priceText = item.price > 0 ? `₹${{item.price.toLocaleString()}}` : '';
                        let typeClass = `type-${{item.type.toLowerCase().replace(/[^a-z]/g, '')}}`;
                        
                        let hintText = '';
                        if (query.length === 1 && (item.type === 'Brand' || item.type === 'Category')) {{
                            hintText = '<span class="single-letter-hint">Try it!</span>';
                        }}
                        
                        div.innerHTML = `
                            <div class="suggestion-main">
                                <div class="suggestion-text">${{item.suggestion}}</div>
                                <div class="suggestion-scope">
                                    <span>${{item.scope}}</span>
                                </div>
                            </div>
                            <div class="suggestion-meta">
                                <span class="suggestion-type ${{typeClass}}">${{item.type}}</span>
                                ${{priceText ? `<div class="price">${{priceText}}</div>` : ''}}
                            </div>
                            ${{hintText}}
                        `;
                        
                        div.onclick = () => {{
                            searchInput.value = item.suggestion;
                            hideSuggestions();
                            // Trigger another search if it's a brand/category
                            if (item.type === 'Brand' || item.type === 'Category') {{
                                setTimeout(() => fetchSuggestions(item.suggestion), 100);
                            }}
                        }};
                        
                        div.addEventListener('mouseover', () => {{
                            selectedIndex = index;
                            updateSelection(suggestionsContainer.children);
                        }});
                        
                        suggestionsContainer.appendChild(div);
                    }});
                    showSuggestions();
                }} else {{
                    hideSuggestions();
                }}
            }})
            .catch(error => {{
                console.error('Search error:', error);
                showError();
                setTimeout(hideSuggestions, 3000);
            }});
    }}

    function showSuggestions() {{
        if (suggestionsContainer.children.length > 0) {{
            suggestionsContainer.style.display = 'block';
        }}
    }}

    function hideSuggestions() {{
        suggestionsContainer.style.display = 'none';
        hideLoading();
        errorDiv.style.display = 'none';
        selectedIndex = -1;
    }}

    // Enhanced keyboard navigation
    searchInput.addEventListener('keydown', function(e) {{
        const items = suggestionsContainer.children;
        if (items.length === 0) return;

        if (e.key === 'ArrowDown') {{
            e.preventDefault();
            selectedIndex = (selectedIndex + 1) % items.length;
            updateSelection(items);
        }} else if (e.key === 'ArrowUp') {{
            e.preventDefault();
            selectedIndex = (selectedIndex - 1 + items.length) % items.length;
            updateSelection(items);
        }} else if (e.key === 'Enter') {{
            e.preventDefault();
            if (selectedIndex > -1 && items[selectedIndex]) {{
                items[selectedIndex].click();
            }}
        }} else if (e.key === 'Escape') {{
            hideSuggestions();
        }}
    }});

    function updateSelection(items) {{
        for (let i = 0; i < items.length; i++) {{
            if (i === selectedIndex) {{
                items[i].classList.add('selected');
                items[i].scrollIntoView({{ block: 'nearest' }});
            }} else {{
                items[i].classList.remove('selected');
            }}
        }}
    }}

    // Click outside to close
    document.addEventListener('click', function(event) {{
        if (!searchInput.contains(event.target) && !suggestionsContainer.contains(event.target)) {{
            hideSuggestions();
        }}
    }});

    searchInput.addEventListener('focus', function() {{
        if (this.value.length >= 1) {{
            fetchSuggestions(this.value);
        }}
    }});
    </script>
    """

    components.html(search_html, height=500)
    
    # Show popular brands and categories with FIXED st.write usage
    col1, col2 = st.columns(2)
    
    with col1:
        if st.expander("🏷️ Brands by Letter", expanded=False):
            if brand_index:
                for letter in sorted(brand_index.keys()):
                    brands_for_letter = [b['display_name'] for b in brand_index[letter][:3]]
                    st.write(f"**{letter.upper()}:** {', '.join(brands_for_letter)}")
            else:
                st.write("No brand index available")
    
    with col2:
        if st.expander("📂 Categories by Letter", expanded=False):
            if category_index:
                for letter in sorted(category_index.keys()):
                    categories_for_letter = [c['display_name'] for c in category_index[letter][:3]]
                    st.write(f"**{letter.upper()}:** {', '.join(categories_for_letter)}")
            else:
                st.write("No category index available")
    
    # Debug information
    with st.expander("🔧 Debug Info", expanded=False):
        st.write(f"API Port: {st.session_state.get('api_port', API_PORT)}")
        st.write(f"Brand Index Letters: {sorted(brand_index.keys()) if brand_index else 'None'}")
        st.write(f"Category Index Letters: {sorted(category_index.keys()) if category_index else 'None'}")
        st.write(f"Sample Brands for 'a': {[b['display_name'] for b in brand_index.get('a', [])] if brand_index else 'None'}")
        st.write(f"Sample Categories for 's': {[c['display_name'] for c in category_index.get('s', [])] if category_index else 'None'}")

else:
    st.error("Failed to initialize the application. Please check the CSV file path and format.")
    
# Add a button to restart the API if needed
if st.button("🔄 Restart Search Service"):
    if 'app_initialized' in st.session_state:
        del st.session_state.app_initialized
    if 'api_thread_started' in st.session_state:
        del st.session_state.api_thread_started
    st.rerun()
