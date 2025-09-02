##############################################################################
# Enhanced E-commerce Search with Single Letter Brand/Category Suggestions + Gender Relevance
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

# Global variables for enhanced search
product_data = {}
brand_index = {}  # Single letter -> brands
category_index = {}  # Single letter -> categories
gender_index = {}  # Gender -> products
brand_products = defaultdict(list)
category_products = defaultdict(list)
search_index = {}
popular_brands = []
popular_categories = []

# Gender relevance mapping for categories
CATEGORY_GENDER_MAPPING = {
    'sarees': {'Women'},
    'kurtas': {'Men', 'Women'},
    'lehenga': {'Women'},
    'shirts': {'Men', 'Women'},
    'dresses': {'Women'},
    'suits': {'Men', 'Women'},
    'jeans': {'Men', 'Women'},
    'shoes': {'Men', 'Women'},
    'sandals': {'Men', 'Women'},
    'heels': {'Women'},
    'watches': {'Men', 'Women'},
    'bags': {'Women'},
    'wallets': {'Men', 'Women'},
    't-shirts': {'Men', 'Women'},
    'trousers': {'Men', 'Women'},
    'shorts': {'Men', 'Women'},
    'skirts': {'Women'},
    'blouses': {'Women'},
    'jackets': {'Men', 'Women'},
    'coats': {'Men', 'Women'},
    'sneakers': {'Men', 'Women'},
    'flipflops': {'Men', 'Women'},
    'caps': {'Men', 'Women'},
    'hats': {'Men', 'Women'},
    'belts': {'Men', 'Women'},
    'scarves': {'Men', 'Women'},
    'gloves': {'Men', 'Women'},
    'socks': {'Men', 'Women'},
    'ties': {'Men'},
    'cufflinks': {'Men'},
    'lingerie': {'Women'},
    'innerwear': {'Men', 'Women'},
    'nightwear': {'Men', 'Women'},
    'sportswear': {'Men', 'Women'},
    'swimwear': {'Men', 'Women'},
    'backpacks': {'Men', 'Women'},
    'jewelry': {'Women'},
    'earrings': {'Women'},
    'necklaces': {'Women'},
    'bracelets': {'Women'},
    'sunglasses': {'Men', 'Women'},
    'perfumes': {'Men', 'Women'},
    'formal shoes': {'Men'},
    'casual shoes': {'Men', 'Women'},
    'loafers': {'Men', 'Women'},
    'tops': {'Women'},
    'ethnic': {'Men', 'Women'},
    'makeup': {'Women'},
    'others': {'Men', 'Women'}
}

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

def normalize_gender(gender):
    """Normalize gender values"""
    if not gender or pd.isna(gender):
        return 'Unisex'
    
    gender = str(gender).lower().strip()
    if gender in ['men', 'male', 'man', 'boys', 'boy']:
        return 'Men'
    elif gender in ['women', 'female', 'woman', 'girls', 'girl']:
        return 'Women'
    elif gender in ['unisex', 'both', 'all']:
        return 'Unisex'
    else:
        return 'Unisex'

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
        'ethnic': ['lehenga', 'salwar', 'dupatta', 'ethnic', 'traditional'],
        'skirts': ['skirt', 'mini skirt', 'maxi skirt'],
        'ties': ['tie', 'necktie', 'bow tie'],
        'formal shoes': ['formal shoes', 'oxford', 'derby', 'dress shoes'],
        'casual shoes': ['casual shoes', 'canvas shoes', 'slip-on'],
        'heels': ['heels', 'high heels', 'stiletto', 'pumps'],
        'flipflops': ['flip flop', 'flipflop', 'slipper', 'slides']
    }
    
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in product_name_lower:
                return category
    
    return 'others'

def is_gender_relevant(category, product_gender, target_gender=None):
    """Check if a product category is relevant for a specific gender"""
    if not target_gender or target_gender == 'Unisex':
        return True
    
    # Get expected genders for this category
    expected_genders = CATEGORY_GENDER_MAPPING.get(category, {'Men', 'Women'})
    
    # If target gender matches expected genders or product gender
    if target_gender in expected_genders:
        return True
    
    # If product gender matches target gender
    if product_gender == target_gender:
        return True
    
    # If product is unisex, it's relevant for all
    if product_gender == 'Unisex':
        return True
    
    return False

def get_gender_score(category, product_gender, target_gender=None):
    """Get relevance score based on gender matching"""
    if not target_gender or target_gender == 'Unisex':
        return 1.0
    
    expected_genders = CATEGORY_GENDER_MAPPING.get(category, {'Men', 'Women'})
    
    # Perfect match: target gender in expected genders AND matches product gender
    if target_gender in expected_genders and product_gender == target_gender:
        return 1.0
    
    # Good match: target gender in expected genders OR matches product gender
    if target_gender in expected_genders or product_gender == target_gender:
        return 0.9
    
    # Unisex products are always somewhat relevant
    if product_gender == 'Unisex':
        return 0.7
    
    # Category allows target gender but product is for different gender
    if target_gender in expected_genders:
        return 0.5
    
    # Low relevance
    return 0.3

def detect_gender_from_query(query):
    """Detect intended gender from search query"""
    query_lower = clean_text(query)
    
    men_keywords = ['men', 'male', 'man', 'boys', 'boy', 'mens', 'masculine', 'him', 'his', 'guy', 'guys']
    women_keywords = ['women', 'female', 'woman', 'girls', 'girl', 'womens', 'feminine', 'her', 'hers', 'lady', 'ladies']
    
    men_score = sum(1 for word in men_keywords if word in query_lower)
    women_score = sum(1 for word in women_keywords if word in query_lower)
    
    if men_score > women_score:
        return 'Men'
    elif women_score > men_score:
        return 'Women'
    else:
        return None

def build_single_letter_indexes():
    """Build indexes for single letter brand and category suggestions with gender awareness"""
    global brand_index, category_index, gender_index, popular_brands, popular_categories
    
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
                
                # Calculate gender distribution for this brand
                gender_dist = {'Men': 0, 'Women': 0, 'Unisex': 0}
                for product_key in products:
                    if product_key in product_data:
                        gender = product_data[product_key]['gender']
                        gender_dist[gender] = gender_dist.get(gender, 0) + 1
                
                brand_index[first_letter].append({
                    'name': brand,
                    'count': len(products),
                    'display_name': brand.title(),
                    'gender_dist': gender_dist
                })
                brand_count[brand] = len(products)
    
    # Sort brands by popularity within each letter
    for letter in brand_index:
        brand_index[letter].sort(key=lambda x: x['count'], reverse=True)
        brand_index[letter] = brand_index[letter][:6]  # Top 6 per letter
    
    # Build category index (first letter -> categories) with gender relevance
    category_count = Counter()
    valid_categories = set(CATEGORY_GENDER_MAPPING.keys())
    
    for category, products in category_products.items():
        if (category and len(category) > 1 and 
            category.lower() in valid_categories and 
            len(products) >= 10):  # Only categories with at least 10 products
            
            first_letter = category[0].lower()
            if first_letter.isalpha():  # Only alphabetic letters
                if first_letter not in category_index:
                    category_index[first_letter] = []
                
                # Calculate gender distribution for this category
                gender_dist = {'Men': 0, 'Women': 0, 'Unisex': 0}
                for product_key in products:
                    if product_key in product_data:
                        gender = product_data[product_key]['gender']
                        gender_dist[gender] = gender_dist.get(gender, 0) + 1
                
                expected_genders = CATEGORY_GENDER_MAPPING.get(category, {'Men', 'Women'})
                
                category_index[first_letter].append({
                    'name': category,
                    'count': len(products),
                    'display_name': category.title(),
                    'gender_dist': gender_dist,
                    'expected_genders': expected_genders
                })
                category_count[category] = len(products)
    
    # Sort categories by popularity within each letter
    for letter in category_index:
        category_index[letter].sort(key=lambda x: x['count'], reverse=True)
        category_index[letter] = category_index[letter][:5]  # Top 5 per letter
    
    # Build gender index
    for product_key, product in product_data.items():
        gender = product['gender']
        if gender not in gender_index:
            gender_index[gender] = []
        gender_index[gender].append(product_key)
    
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
        st.info(f"Processing {len(df)} products for enhanced search with gender relevance...")
        
        processed_count = 0
        
        for idx, row in df.iterrows():
            try:
                product_name = str(row['product_name']).strip()
                brand = str(row['ProductBrand']).strip()
                gender = normalize_gender(row['Gender'])
                price = row['price']
                
                if not product_name or product_name.lower() == 'nan':
                    continue
                
                clean_name = clean_text(product_name)
                category = extract_category_from_name(product_name)
                
                if not brand or brand.lower() == 'nan':
                    brand = 'Unknown'
                
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
    """Enhanced suggestions with single letter brand/category support and gender relevance"""
    if not query:
        return []
    
    original_query = query
    query_clean = clean_text(query)
    suggestions = []
    seen = set()
    
    # Detect intended gender from query
    detected_gender = detect_gender_from_query(original_query)
    
    # Special handling for single character queries - ONLY brands and categories
    if len(original_query) == 1:
        letter = original_query.lower()
        
        # Add brand suggestions for this letter (higher priority)
        if letter in brand_index:
            for brand_info in brand_index[letter][:5]:  # Top 5 brands
                # Add gender info to scope
                gender_info = brand_info['gender_dist']
                dominant_gender = max(gender_info.keys(), key=lambda k: gender_info[k])
                scope_text = f"{brand_info['count']} products"
                if detected_gender and gender_info.get(detected_gender, 0) > 0:
                    scope_text += f" • {gender_info[detected_gender]} for {detected_gender}"
                elif dominant_gender != 'Unisex':
                    scope_text += f" • Mainly {dominant_gender}"
                
                suggestions.append({
                    'suggestion': brand_info['display_name'],
                    'type': 'Brand',
                    'scope': scope_text,
                    'category': 'brand',
                    'gender': dominant_gender,
                    'price': 0,
                    '_score': 100 + brand_info['count'] * 0.1
                })
        
        # Add category suggestions for this letter
        if letter in category_index:
            for cat_info in category_index[letter][:5]:  # Top 5 categories
                expected_genders = cat_info['expected_genders']
                gender_relevance = ""
                if len(expected_genders) == 1:
                    gender_relevance = f" • {list(expected_genders)[0]} only"
                elif detected_gender and detected_gender in expected_genders:
                    gender_relevance = f" • Perfect for {detected_gender}"
                
                scope_text = f"{cat_info['count']} items{gender_relevance}"
                
                # Boost score if gender matches
                score_boost = 0
                if detected_gender and detected_gender in expected_genders:
                    score_boost = 10
                
                suggestions.append({
                    'suggestion': cat_info['display_name'],
                    'type': 'Category',
                    'scope': scope_text,
                    'category': cat_info['name'],
                    'gender': 'All',
                    'price': 0,
                    '_score': 95 + cat_info['count'] * 0.1 + score_boost
                })
        
        # Sort by score and return ONLY brands and categories for single letter
        suggestions.sort(key=lambda x: x['_score'], reverse=True)
        return suggestions[:limit]
    
    # For multi-character queries, use existing logic with gender enhancements
    
    # 1. Exact matches with gender scoring
    if query_clean in search_index:
        matches = search_index[query_clean]
        if isinstance(matches, str):
            matches = [matches]
        
        for match in matches[:3]:
            if match in product_data and match not in seen:
                product = product_data[match]
                gender_score = get_gender_score(product['category'], product['gender'], detected_gender)
                
                scope_text = f"{product['brand']} • {product['gender']}"
                if detected_gender and gender_score > 0.8:
                    scope_text += " ✓"
                
                suggestions.append({
                    'suggestion': product['original_name'],
                    'type': 'Exact Match',
                    'scope': scope_text,
                    'category': product['category'],
                    'gender': product['gender'],
                    'price': product['price'],
                    '_score': 100 * gender_score
                })
                seen.add(match)
    
    # 2. Brand matching (enhanced with gender info)
    for brand in brand_products.keys():
        if query_clean in brand or brand.startswith(query_clean):
            brand_display = brand.title()
            if brand_display not in [s['suggestion'] for s in suggestions]:
                # Calculate gender distribution for brand
                brand_products_list = brand_products[brand]
                gender_dist = {'Men': 0, 'Women': 0, 'Unisex': 0}
                for prod_key in brand_products_list:
                    if prod_key in product_data:
                        gender = product_data[prod_key]['gender']
                        gender_dist[gender] = gender_dist.get(gender, 0) + 1
                
                dominant_gender = max(gender_dist.keys(), key=lambda k: gender_dist[k])
                scope_text = f"{len(brand_products_list)} products"
                if detected_gender and gender_dist.get(detected_gender, 0) > 0:
                    scope_text += f" • {gender_dist[detected_gender]} for {detected_gender}"
                
                suggestions.append({
                    'suggestion': brand_display,
                    'type': 'Brand',
                    'scope': scope_text,
                    'category': 'brand',
                    'gender': dominant_gender,
                    'price': 0,
                    '_score': 95
                })
    
    # 3. Category matching (enhanced with gender relevance)
    for category in category_products.keys():
        if query_clean in category or category.startswith(query_clean):
            category_display = category.title()
            if category_display not in [s['suggestion'] for s in suggestions]:
                expected_genders = CATEGORY_GENDER_MAPPING.get(category, {'Men', 'Women'})
                gender_relevance = ""
                score_boost = 0
                
                if len(expected_genders) == 1:
                    gender_relevance = f" • {list(expected_genders)[0]} only"
                elif detected_gender and detected_gender in expected_genders:
                    gender_relevance = f" • Perfect for {detected_gender}"
                    score_boost = 8
                
                scope_text = f"{len(category_products[category])} items{gender_relevance}"
                
                suggestions.append({
                    'suggestion': category_display,
                    'type': 'Category',
                    'scope': scope_text,
                    'category': category,
                    'gender': 'All',
                    'price': 0,
                    '_score': 92 + score_boost
                })
    
    # 4. Prefix matches with gender filtering
    for key, matches in search_index.items():
        if key.startswith(query_clean) and len(suggestions) < limit * 2:
            if isinstance(matches, str):
                matches = [matches]
            
            # Sort matches by gender relevance
            gender_sorted_matches = []
            for match in matches[:5]:
                if match in product_data:
                    product = product_data[match]
                    gender_score = get_gender_score(product['category'], product['gender'], detected_gender)
                    gender_sorted_matches.append((match, gender_score))
            
            gender_sorted_matches.sort(key=lambda x: x[1], reverse=True)
            
            for match, gender_score in gender_sorted_matches[:2]:
                if match not in seen:
                    product = product_data[match]
                    scope_text = f"{product['brand']} • {product['gender']}"
                    if detected_gender and gender_score > 0.8:
                        scope_text += " ✓"
                    
                    suggestions.append({
                        'suggestion': product['original_name'],
                        'type': 'Autocomplete',
                        'scope': scope_text,
                        'category': product['category'],
                        'gender': product['gender'],
                        'price': product['price'],
                        '_score': 88 * gender_score
                    })
                    seen.add(match)
    
    # 5. Fuzzy matching with gender preference (for 2+ characters)
    if len(query_clean) >= 2 and len(suggestions) < limit:
        all_product_names = [(data['original_name'], key) for key, data in product_data.items()]
        product_names_only = [name for name, _ in all_product_names]
        
        fuzzy_matches = process.extract(
            original_query, 
            product_names_only, 
            scorer=fuzz.partial_ratio, 
            limit=limit * 2
        )
        
        # Sort fuzzy matches by gender relevance
        gender_scored_matches = []
        for match_name, score, _ in fuzzy_matches:
            if score > 65:
                matching_key = None
                for orig_name, key in all_product_names:
                    if orig_name == match_name:
                        matching_key = key
                        break
                
                if matching_key and matching_key not in seen and matching_key in product_data:
                    product = product_data[matching_key]
                    gender_score = get_gender_score(product['category'], product['gender'], detected_gender)
                    combined_score = score * 0.8 * gender_score
                    gender_scored_matches.append((matching_key, product, combined_score))
        
        gender_scored_matches.sort(key=lambda x: x[2], reverse=True)
        
        for matching_key, product, combined_score in gender_scored_matches:
            scope_text = f"{product['brand']} • {product['gender']}"
            if detected_gender and get_gender_score(product['category'], product['gender'], detected_gender) > 0.8:
                scope_text += " ✓"
            
            suggestions.append({
                'suggestion': product['original_name'],
                'type': 'Similar',
                'scope': scope_text,
                'category': product['category'],
                'gender': product['gender'],
                'price': product['price'],
                '_score': combined_score
            })
            seen.add(matching_key)
            
            if len(suggestions) >= limit:
                break
    
    # Sort by score and return
    suggestions.sort(key=lambda x: x['_score'], reverse=True)
    return suggestions[:limit]

@api_app.get("/suggestions")
async def suggestions_endpoint(q: str = Query(..., min_length=1), limit: int = 10):
    """API endpoint for enhanced search suggestions with gender relevance"""
    return get_smart_suggestions(q, limit)

@api_app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "products_loaded": len(product_data),
        "brands_indexed": len(brand_products),
        "categories_indexed": len(category_products),
        "search_terms": len(search_index),
        "gender_categories": len([k for k, v in CATEGORY_GENDER_MAPPING.items() if len(v) == 1]),
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
        with st.spinner("Loading enhanced product search with gender relevance..."):
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
st.title("🛍️ Enhanced Product Search with Gender Intelligence")
st.markdown("*Type even a single letter to see brands and categories! Add 'men' or 'women' to get targeted suggestions.*")

if initialize_app():
    # Show enhanced statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Products", len(product_data))
    
    with col2:
        st.metric("Brands", len(brand_products))
    
    with col3:
        st.metric("Categories", len(category_products))
    
    with col4:
        gender_specific = len([k for k, v in CATEGORY_GENDER_MAPPING.items() if len(v) == 1])
        st.metric("Gender-Specific", gender_specific)
    
    with col5:
        st.metric("API Port", st.session_state.get('api_port', API_PORT))
    
    # Enhanced search interface with gender awareness
    search_html = f"""
    <div style="position: relative; display: inline-block; width: 100%;">
        <input type="text" id="searchInput" placeholder="Try: 'n' for Nike, 's' for shirts, 'women jeans', 'men watches'..." 
               style="width:100%; max-width:800px; font-size:18px; padding:16px; border: 3px solid #4CAF50; 
                      border-radius: 12px; outline: none; box-shadow: 0 3px 10px rgba(76,175,80,0.3);
                      transition: all 0.2s ease;"/>
        <div id="suggestions-container" style="
            border: 1px solid #ddd;
            max-height: 450px; 
            width: 100%;
            max-width: 800px;
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
            ⚡ Searching with gender intelligence...
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
    
    .gender-hint {{
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
    
    .gender-match {{
        color: #4CAF50 !important;
        font-weight: bold;
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
                        
                        // Check for gender match indicator
                        let scopeClass = item.scope.includes('✓') ? 'gender-match' : '';
                        
                        let hintText = '';
                        if (query.length === 1 && (item.type === 'Brand' || item.type === 'Category')) {{
                            hintText = '<span class="gender-hint">Try it!</span>';
                        }}
                        
                        div.innerHTML = `
                            <div class="suggestion-main">
                                <div class="suggestion-text">${{item.suggestion}}</div>
                                <div class="suggestion-scope ${{scopeClass}}">
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
    
    # Show gender-aware statistics and examples
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
                    categories_for_letter = []
                    for c in category_index[letter][:3]:
                        expected_genders = c.get('expected_genders', {'Men', 'Women'})
                        if len(expected_genders) == 1:
                            gender_tag = f" ({list(expected_genders)[0]})"
                        else:
                            gender_tag = ""
                        categories_for_letter.append(c['display_name'] + gender_tag)
                    st.write(f"**{letter.upper()}:** {', '.join(categories_for_letter)}")
            else:
                st.write("No category index available")
    
    # Gender examples section
    with st.expander("🎯 Gender-Smart Search Examples", expanded=False):
        st.markdown("""
        **Try these gender-aware searches:**
        - `women jeans` - Shows jeans with preference for women's styles
        - `men watches` - Prioritizes men's watch collections
        - `s` - Shows shirts (unisex) and sarees (women-only)
        - `h` - Shows handbags (women) and hats (unisex)
        - `nike women` - Nike products filtered for women
        - `formal shoes men` - Men's formal footwear
        
        **Gender-specific categories:**
        - Women only: Sarees, Dresses, Heels, Handbags, Jewelry, Makeup
        - Men only: Ties, Cufflinks, Formal Shoes
        - Unisex: Jeans, T-shirts, Watches, Sunglasses, Sneakers
        """)
    
    # Debug information
    with st.expander("🔧 Debug Info", expanded=False):
        st.write(f"API Port: {st.session_state.get('api_port', API_PORT)}")
        st.write(f"Brand Index Letters: {sorted(brand_index.keys()) if brand_index else 'None'}")
        st.write(f"Category Index Letters: {sorted(category_index.keys()) if category_index else 'None'}")
        st.write(f"Gender Distribution: {dict(Counter(p['gender'] for p in product_data.values()))}")
        st.write(f"Sample Gender-Specific Categories: {[(k, list(v)) for k, v in list(CATEGORY_GENDER_MAPPING.items())[:5] if len(v) == 1]}")

else:
    st.error("Failed to initialize the application. Please check the CSV file path and format.")

# Add a button to restart the API if needed
if st.button("🔄 Restart Search Service"):
    if 'app_initialized' in st.session_state:
        del st.session_state.app_initialized
    if 'api_thread_started' in st.session_state:
        del st.session_state.api_thread_started
    st.rerun()
