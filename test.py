


# ##############################################################################
# # Professional E-commerce Search with Gender-Category Aware Autocomplete
# import streamlit as st
# import streamlit.components.v1 as components
# import pandas as pd
# import threading
# from fastapi import FastAPI, Query
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import nest_asyncio
# from fast_autocomplete import AutoComplete
# from rapidfuzz import fuzz, process
# import json
# import os
# import warnings
# from collections import defaultdict
# from rapidfuzz import fuzz, process

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

# # Global variables
# autocomplete_instance = None
# product_database = defaultdict(lambda: defaultdict(list))  # {category: {gender: [products]}}
# category_gender_mapping = defaultdict(set)  # {category: {valid_genders}}

# def initialize_autocomplete():
#     """Initialize Fast-Autocomplete with enhanced product categorization"""
#     global autocomplete_instance, product_database, category_gender_mapping
    
#     if autocomplete_instance is not None:
#         return autocomplete_instance
    
#     # Base words with proper categorization
#     words = {}
    
#     # Gender-specific categories mapping
#     category_rules = {
#     'saree': {'Women'},
#     'kurta': {'Men', 'Women'},
#     'lehenga': {'Women'},
#     'shirt': {'Men', 'Women'},
#     'dress': {'Women'},
#     'suit': {'Men', 'Women'},
#     'jeans': {'Men', 'Women'},
#     'shoes': {'Men', 'Women'},
#     'sandals': {'Men', 'Women'},
#     'heels': {'Women'},
#     'watch': {'Men', 'Women'},
#     'handbag': {'Women'},
#     'wallet': {'Men', 'Women'},
#     't-shirt': {'Men', 'Women'},
#     'trousers': {'Men', 'Women'},
#     'shorts': {'Men', 'Women'},
#     'skirt': {'Women'},
#     'blouse': {'Women'},
#     'jackets': {'Men', 'Women'},
#     'coats': {'Men', 'Women'},
#     'sneakers': {'Men', 'Women'},
#     'flipflops': {'Men', 'Women'},
#     'cap': {'Men', 'Women'},
#     'hat': {'Men', 'Women'},
#     'belt': {'Men', 'Women'},
#     'scarf': {'Men', 'Women'},
#     'gloves': {'Men', 'Women'},
#     'socks': {'Men', 'Women'},
#     'tie': {'Men'},
#     'cufflinks': {'Men'},
#     'heels': {'Women'},
#     'lingerie': {'Women'},
#     'nightwear': {'Men', 'Women'},
#     'sportswear': {'Men', 'Women'},
#     'swimwear': {'Men', 'Women'},
#     'backpack': {'Men', 'Women'},
#     'jewelry': {'Women'},
#     'earrings': {'Women'},
#     'necklace': {'Women'},
#     'bracelet': {'Women'},
#     'sunglasses': {'Men', 'Women'},
#     'perfume': {'Men', 'Women'},
#     'watch': {'Men', 'Women'},
#     'formal shoes': {'Men'},
#     'casual shoes': {'Men', 'Women'},
#     'loafers': {'Men', 'Women'},
# }

    
#     csv_path = "/home/artisans15/projects/fashion_retail_analytics/data/raw/myntra_products_catalog.csv"
#     if os.path.exists(csv_path):
#         try:
#             df = pd.read_csv(csv_path).head(20000)

#             for _, row in df.iterrows():
#                 gender = str(row.get('Gender', 'Unisex')).strip()
#                 if gender.lower() in ['nan', 'unisex', '']:
#                     gender = 'Unisex'
#                 brand = str(row.get('ProductBrand', '')).strip()
#                 product_name = str(row.get('product_name', '')).strip()
#                 category = str(row.get('Category', '')).strip()
#                 # Only add to autocomplete if present in dataset
#                 if product_name and len(product_name) > 1:
#                     key = product_name.lower()
#                     words[key] = {
#                         'context': {
#                             'type': 'product',
#                             'category': category,
#                             'valid_genders': [gender],
#                             'brand': brand
#                         }
#                     }
#                 # Add n-grams (phrases) from product name for better phrase search
#                 tokens = product_name.lower().split()
#                 for n in range(2, min(5, len(tokens)+1)):  # n-grams from 2 to 4 words
#                     for i in range(len(tokens)-n+1):
#                         phrase = " ".join(tokens[i:i+n])
#                         if phrase not in words:
#                             words[phrase] = {
#                                 'context': {
#                                     'type': 'phrase',
#                                     'category': category,
#                                     'valid_genders': [gender],
#                                     'brand': brand
#                                 }
#                             }

#                 if brand and len(brand) > 1:
#                     brand_key = brand.lower()
#                     if brand_key not in words:
#                         words[brand_key] = {
#                             'context': {
#                                 'type': 'brand',
#                                 'category': category,
#                                 'valid_genders': [gender]
#                             }
#                         }
#                     else:
#                         if gender not in words[brand_key]['context']['valid_genders']:
#                             words[brand_key]['context']['valid_genders'].append(gender)

                
#                 # Process product name and categorize
#                 product_words = product_name.split()
#                 for word in product_words:
#                     clean_word = word.strip('.,!?-')
#                     if len(clean_word) > 2:
#                         # Determine category from word
#                         word_category = None
#                         for cat, valid_genders in category_rules.items():
#                             if cat in clean_word:
#                                 word_category = cat
#                                 break
                        
#                         if not word_category:
#                             word_category = category.lower() if category else 'general'
                        
#                         if clean_word not in words:
#                             words[clean_word] = {
#                                 'context': {
#                                     'type': 'product',
#                                     'category': word_category,
#                                     'valid_genders': [gender]
#                                 }
#                             }
#                         else:
#                             # Add gender to existing product
#                             if gender not in words[clean_word]['context']['valid_genders']:
#                                 words[clean_word]['context']['valid_genders'].append(gender)
                        
#                         # Store in product database
#                         product_database[word_category][gender].append({
#                             'name': product_name,
#                             'brand': brand,
#                             'word': clean_word
#                         })
                        
#                         # Update category-gender mapping
#                         category_gender_mapping[word_category].add(gender)
            
#             st.success(f"Loaded {len(df)} products with gender-category mapping.")
#         except Exception as e:
#             st.warning(f"Could not load CSV: {e}. Using sample data.")
#             # Add sample data with proper categorization
#             sample_data = {
#                 'saree': {'context': {'type': 'product', 'category': 'saree', 'valid_genders': ['Women']}},
#                 'kurta': {'context': {'type': 'product', 'category': 'kurta', 'valid_genders': ['Men', 'Women']}},
#                 'shirt': {'context': {'type': 'product', 'category': 'shirt', 'valid_genders': ['Men', 'Women']}},
#                 'dress': {'context': {'type': 'product', 'category': 'dress', 'valid_genders': ['Women']}},
#                 'jeans': {'context': {'type': 'product', 'category': 'jeans', 'valid_genders': ['Men', 'Women']}},
#                 'lehenga': {'context': {'type': 'product', 'category': 'lehenga', 'valid_genders': ['Women']}},
#                 'shoes': {'context': {'type': 'product', 'category': 'shoes', 'valid_genders': ['Men', 'Women']}},
#                 'sandals': {'context': {'type': 'product', 'category': 'sandals', 'valid_genders': ['Men', 'Women']}},
#                 'heels': {'context': {'type': 'product', 'category': 'heels', 'valid_genders': ['Women']}},
#                 'watch': {'context': {'type': 'product', 'category': 'watch', 'valid_genders': ['Men', 'Women']}},
#                 'handbag': {'context': {'type': 'product', 'category': 'handbag', 'valid_genders': ['Women']}},
#                 'wallet': {'context': {'type': 'product', 'category': 'wallet', 'valid_genders': ['Men', 'Women']}},
#                 't-shirt': {'context': {'type': 'product', 'category': 't-shirt', 'valid_genders': ['Men', 'Women']}},
#                 'jacket': {'context': {'type': 'product', 'category': 'jackets', 'valid_genders': ['Men', 'Women']}},
#                 'skirt': {'context': {'type': 'product', 'category': 'skirt', 'valid_genders': ['Women']}},
#                 'blouse': {'context': {'type': 'product', 'category': 'blouse', 'valid_genders': ['Women']}},
#                 'backpack': {'context': {'type': 'product', 'category': 'backpack', 'valid_genders': ['Men', 'Women']}},
#                 'necklace': {'context': {'type': 'product', 'category': 'necklace', 'valid_genders': ['Women']}},
#                 'earrings': {'context': {'type': 'product', 'category': 'earrings', 'valid_genders': ['Women']}},
#                 'belt': {'context': {'type': 'product', 'category': 'belt', 'valid_genders': ['Men', 'Women']}},
#                 'cap': {'context': {'type': 'product', 'category': 'cap', 'valid_genders': ['Men', 'Women']}},
#                 'gloves': {'context': {'type': 'product', 'category': 'gloves', 'valid_genders': ['Men', 'Women']}},
#                 'scarf': {'context': {'type': 'product', 'category': 'scarf', 'valid_genders': ['Men', 'Women']}},
#                 'socks': {'context': {'type': 'product', 'category': 'socks', 'valid_genders': ['Men', 'Women']}},
#                 'tie': {'context': {'type': 'product', 'category': 'tie', 'valid_genders': ['Men']}},
#                 'cufflinks': {'context': {'type': 'product', 'category': 'cufflinks', 'valid_genders': ['Men']}},
#                 'lingerie': {'context': {'type': 'product', 'category': 'lingerie', 'valid_genders': ['Women']}},
#                 'nightwear': {'context': {'type': 'product', 'category': 'nightwear', 'valid_genders': ['Men', 'Women']}},
#                 'sportswear': {'context': {'type': 'product', 'category': 'sportswear', 'valid_genders': ['Men', 'Women']}},
#                 'swimwear': {'context': {'type': 'product', 'category': 'swimwear', 'valid_genders': ['Men', 'Women']}},
#                 'formal shoes': {'context': {'type': 'product', 'category': 'formal shoes', 'valid_genders': ['Men']}},
#                 'casual shoes': {'context': {'type': 'product', 'category': 'casual shoes', 'valid_genders': ['Men', 'Women']}},
#                 'loafers': {'context': {'type': 'product', 'category': 'loafers', 'valid_genders': ['Men', 'Women']}},
#                 'earphones': {'context': {'type': 'product', 'category': 'earphones', 'valid_genders': ['Men', 'Women']}},
#                 'sunglasses': {'context': {'type': 'product', 'category': 'sunglasses', 'valid_genders': ['Men', 'Women']}},
#                 'perfume': {'context': {'type': 'product', 'category': 'perfume', 'valid_genders': ['Men', 'Women']}},
#                 'hoodie': {'context': {'type': 'product', 'category': 'hoodie', 'valid_genders': ['Men', 'Women']}},
#                 'sweater': {'context': {'type': 'product', 'category': 'sweater', 'valid_genders': ['Men', 'Women']}},
#             }

#             words.update(sample_data)
    
#     # Create autocomplete instance
#     autocomplete_instance = AutoComplete(words=words)
#     return autocomplete_instance

# all_product_names = []
# if os.path.exists("/home/artisans15/projects/fashion_retail_analytics/data/raw/myntra_products_catalog.csv"):
#     try:
#         df = pd.read_csv("/home/artisans15/projects/fashion_retail_analytics/data/raw/myntra_products_catalog.csv").head(20000)
#         all_product_names = df['product_name'].dropna().astype(str).str.lower().unique().tolist()
#     except Exception as e:
#         st.warning(f"Could not load product names: {e}")
# def get_smart_suggestions(query: str, limit: int = 10):
#     if not query or len(query) < 1:
#         return []

#     autocomplete = initialize_autocomplete()
#     suggestions_ranked = []
#     seen = set()

#     # --- 1. Full fuzzy matches ---
#     if all_product_names:
#         matches = process.extract(query, all_product_names, scorer=fuzz.token_sort_ratio, limit=limit * 3)
#         for match, score, _ in matches:
#             if score > 80 and match.lower() not in seen:
#                 suggestions_ranked.append((
#                     score + 100,
#                     {
#                         "suggestion": match,
#                         "type": "Product",
#                         "scope": "from Catalog",
#                         "category": "full-product",
#                         "gender": "Unisex"
#                     }
#                 ))
#                 seen.add(match.lower())

#     # --- 2. Phrase-based autocomplete ---
#     raw_suggestions = autocomplete.search(word=query, max_cost=2, size=limit * 3)
#     suggestions_flat = [item for sublist in raw_suggestions for item in sublist]

#     for sug in suggestions_flat:
#         sug_lower = sug.lower()
#         if sug_lower in seen or sug_lower == query.lower():
#             continue

#         context_info = None
#         # Try to get context for the whole phrase first
#         if sug_lower in autocomplete.words:
#             context_info = autocomplete.words[sug_lower]['context']
#         else:
#             # Fallback: check for context in any word in the phrase
#             for word in sug_lower.split():
#                 if word in autocomplete.words:
#                     context_info = autocomplete.words[word]['context']
#                     break
#         if not context_info:
#             continue

#         category = context_info.get('category', 'general')
#         valid_genders = context_info.get('valid_genders', ['Unisex'])
#         product_type = context_info.get('type', 'product')

#         for gender in valid_genders:
#             suggestions_ranked.append((
#                 70 if product_type == 'phrase' else 50,
#                 {
#                     "suggestion": sug,
#                     "type": product_type.capitalize(),
#                     "scope": f"for {gender}" if gender != 'Unisex' else "for Everyone",
#                     "category": category,
#                     "gender": gender
#                 }
#             ))
#         seen.add(sug_lower)

#     # --- 3. Sort by score ---
#     suggestions_ranked.sort(key=lambda x: x[0], reverse=True)
#     final_suggestions = [s[1] for s in suggestions_ranked[:limit]]

#     return final_suggestions
##############################################################################
# Professional E-commerce Search with Amazon/Myntra-level Autocomplete
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import threading
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import nest_asyncio
from fast_autocomplete import AutoComplete
from rapidfuzz import fuzz, process
import json
import os
import warnings
from collections import defaultdict, Counter
import re
from rapidfuzz import fuzz, process

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

# Enhanced global variables for sophisticated search
autocomplete_instance = None
product_database = defaultdict(lambda: defaultdict(list))
category_gender_mapping = defaultdict(set)
brand_frequency = Counter()
product_popularity = defaultdict(int)
search_patterns = defaultdict(int)
category_keywords = {}
brand_products = defaultdict(list)
price_ranges = defaultdict(list)

def preprocess_query(query):
    """Advanced query preprocessing like Amazon/Myntra"""
    # Remove common stop words for fashion
    stop_words = {'for', 'with', 'in', 'on', 'the', 'a', 'an', 'and', 'or'}
    
    # Normalize query
    query = query.lower().strip()
    query = re.sub(r'[^\w\s]', ' ', query)  # Remove punctuation
    query = ' '.join([word for word in query.split() if word not in stop_words])
    
    return query

def extract_intent(query):
    """Extract search intent like Amazon does"""
    intents = {
        'brand': ['nike', 'adidas', 'puma', 'reebok', 'levis', 'calvin klein', 'tommy hilfiger', 'zara', 'h&m'],
        'color': ['red', 'blue', 'green', 'black', 'white', 'pink', 'yellow', 'purple', 'orange', 'brown', 'grey', 'gray'],
        'gender': ['men', 'women', 'mens', 'womens', 'male', 'female', 'boys', 'girls'],
        'occasion': ['casual', 'formal', 'party', 'wedding', 'office', 'gym', 'sports', 'travel'],
        'price': ['cheap', 'affordable', 'expensive', 'budget', 'premium', 'luxury'],
        'size': ['small', 'medium', 'large', 'xl', 'xxl', 's', 'm', 'l']
    }
    
    detected_intents = {}
    words = query.lower().split()
    
    for intent_type, keywords in intents.items():
        for word in words:
            if any(keyword in word for keyword in keywords):
                detected_intents[intent_type] = word
                break
    
    return detected_intents

def calculate_relevance_score(suggestion, query, context, search_history=None):
    """Advanced relevance scoring similar to Amazon's algorithm"""
    score = 0
    
    # 1. Exact match bonus (highest priority)
    if suggestion.lower() == query.lower():
        score += 100
    
    # 2. Prefix match bonus
    if suggestion.lower().startswith(query.lower()):
        score += 80
    
    # 3. Token-based fuzzy matching
    suggestion_tokens = suggestion.lower().split()
    query_tokens = query.lower().split()
    
    token_scores = []
    for qt in query_tokens:
        best_token_score = 0
        for st in suggestion_tokens:
            token_score = fuzz.ratio(qt, st)
            best_token_score = max(best_token_score, token_score)
        token_scores.append(best_token_score)
    
    if token_scores:
        score += sum(token_scores) / len(token_scores) * 0.6
    
    # 4. Popularity boost (simulate click-through rates)
    popularity = product_popularity.get(suggestion.lower(), 0)
    score += min(popularity * 0.1, 20)  # Cap at 20 points
    
    # 5. Brand recognition boost
    for brand in brand_frequency.most_common(50):  # Top 50 brands
        if brand[0].lower() in suggestion.lower():
            score += min(brand[1] * 0.05, 15)  # Cap at 15 points
            break
    
    # 6. Context relevance
    if context:
        # Gender appropriateness
        intents = extract_intent(query)
        if 'gender' in intents:
            query_gender = 'Women' if intents['gender'] in ['women', 'womens', 'female', 'girls'] else 'Men'
            if query_gender in context.get('valid_genders', []):
                score += 25
        
        # Category relevance
        if context.get('type') == 'product':
            score += 20
        elif context.get('type') == 'brand':
            score += 15
        elif context.get('type') == 'phrase':
            score += 10
    
    # 7. Query length consideration (longer queries need higher precision)
    if len(query.split()) > 1:
        if len(set(query.lower().split()) & set(suggestion.lower().split())) >= len(query.split()) * 0.7:
            score += 30
    
    return score

def group_and_deduplicate_suggestions(suggestions, query):
    """Advanced grouping and deduplication like e-commerce platforms"""
    grouped = defaultdict(list)
    seen_suggestions = set()
    
    # Group by category and gender
    for sug in suggestions:
        key = f"{sug['category']}_{sug['gender']}"
        suggestion_lower = sug['suggestion'].lower()
        
        # Advanced deduplication
        if suggestion_lower in seen_suggestions:
            continue
            
        # Check for very similar suggestions (> 90% similarity)
        is_duplicate = False
        for seen in seen_suggestions:
            if fuzz.ratio(suggestion_lower, seen) > 90:
                is_duplicate = True
                break
        
        if not is_duplicate:
            grouped[key].append(sug)
            seen_suggestions.add(suggestion_lower)
    
    # Flatten and prioritize diverse results
    final_suggestions = []
    max_per_group = 2  # Limit per category-gender combination
    
    # Sort groups by relevance
    sorted_groups = sorted(grouped.items(), 
                          key=lambda x: max(s.get('_score', 0) for s in x[1]), 
                          reverse=True)
    
    for group_key, group_suggestions in sorted_groups:
        # Sort within group and take top suggestions
        group_suggestions.sort(key=lambda x: x.get('_score', 0), reverse=True)
        final_suggestions.extend(group_suggestions[:max_per_group])
        
        if len(final_suggestions) >= 10:  # Limit total suggestions
            break
    
    return final_suggestions

def get_contextual_suggestions(query, category_hint=None, gender_hint=None):
    """Generate contextual suggestions based on partial matches"""
    suggestions = []
    
    # Smart category detection from query
    detected_category = None
    for cat in category_keywords:
        if any(keyword in query.lower() for keyword in category_keywords[cat]):
            detected_category = cat
            break
    
    # If category detected, suggest related items
    if detected_category and detected_category in product_database:
        for gender, products in product_database[detected_category].items():
            if gender_hint and gender != gender_hint:
                continue
                
            for product in products[:3]:  # Top 3 per gender
                suggestions.append({
                    'suggestion': product['name'],
                    'type': 'Related Product',
                    'scope': f'in {detected_category} for {gender}',
                    'category': detected_category,
                    'gender': gender,
                    '_score': 60  # Medium priority
                })
    
    return suggestions

# def initialize_autocomplete():
#     """Enhanced initialization with sophisticated categorization"""
#     global autocomplete_instance, product_database, category_gender_mapping
#     global brand_frequency, product_popularity, category_keywords, brand_products, price_ranges
    
#     if autocomplete_instance is not None:
#         return autocomplete_instance
    
#     words = {}
    
#     # Enhanced category rules with keywords
#     category_rules = {
#         'saree': {'Women'},
#         'kurta': {'Men', 'Women'},
#         'lehenga': {'Women'},
#         'shirt': {'Men', 'Women'},
#         'dress': {'Women'},
#         'suit': {'Men', 'Women'},
#         'jeans': {'Men', 'Women'},
#         'shoes': {'Men', 'Women'},
#         'sandals': {'Men', 'Women'},
#         'heels': {'Women'},
#         'watch': {'Men', 'Women'},
#         'handbag': {'Women'},
#         'wallet': {'Men', 'Women'},
#         't-shirt': {'Men', 'Women'},
#         'trousers': {'Men', 'Women'},
#         'shorts': {'Men', 'Women'},
#         'skirt': {'Women'},
#         'blouse': {'Women'},
#         'jackets': {'Men', 'Women'},
#         'coats': {'Men', 'Women'},
#         'sneakers': {'Men', 'Women'},
#         'flipflops': {'Men', 'Women'},
#         'cap': {'Men', 'Women'},
#         'hat': {'Men', 'Women'},
#         'belt': {'Men', 'Women'},
#         'scarf': {'Men', 'Women'},
#         'gloves': {'Men', 'Women'},
#         'socks': {'Men', 'Women'},
#         'tie': {'Men'},
#         'cufflinks': {'Men'},
#         'lingerie': {'Women'},
#         'nightwear': {'Men', 'Women'},
#         'sportswear': {'Men', 'Women'},
#         'swimwear': {'Men', 'Women'},
#         'backpack': {'Men', 'Women'},
#         'jewelry': {'Women'},
#         'earrings': {'Women'},
#         'necklace': {'Women'},
#         'bracelet': {'Women'},
#         'sunglasses': {'Men', 'Women'},
#         'perfume': {'Men', 'Women'},
#         'formal shoes': {'Men'},
#         'casual shoes': {'Men', 'Women'},
#         'loafers': {'Men', 'Women'},
#     }
    
#     # Build category keywords for smart detection
#     category_keywords = {
#     'saree': ['saree', 'sari', 'silk saree', 'cotton saree', 'blouse', 'design', 'printed', 'model', 'wearing'],
#     'kurta': ['kurta', 'kurti', 'ethnic wear', 'straight', 'printed', 'sleeves', 'hem', 'women'],
#     'shirt': ['shirt', 'formal shirt', 'casual shirt', 'sleeves', 't', 'blue', 'casual', 'collar'],
#     'jeans': ['jeans', 'denim', 'pants', 'blue', 'rise', 'look', 'clean', 'mid'],
#     'shoes': ['shoes', 'footwear', 'sneakers', 'boots', 'running', 'pair', 'brand', 'provided', 'puma'],
#     'dress': ['dress', 'gown', 'frock', 'women', 'hem', 'sleeves', 'woven', 'printed'],
#     'watch': ['watch', 'timepiece', 'smartwatch', 'style', 'strap', 'provided', 'brand', 'dialfeatures'],
#     't-shirt': ['t-shirt', 'tee', 't', 'shirt', 'neck', 'round', 'printed'],
#     'bag': ['bag', 'backpack', 'handbag', 'solid', 'zip', 'sling', 'main', '1'],
#     'heels': ['heels', 'stilettos', 'open', 'zip', 'provided', 'brand', 'manufacturer'],
#     'shorts': ['shorts', 'denim shorts', 'regular', 'rise', 'closure', 'solid', 'pockets'],
#     'trousers': ['trousers', 'chinos', 'formal trousers', 'solid', 'closure', 'fit', 'rise', 'regular'],
#     'tops': ['tops', 'blouse', 'pack', 'round', 'neck', 'solid', '3'],
#     'sandals': ['sandals', 'flip-flops', 'pair', 'manufacturer', 'provided', 'brand', 'open'],
#     'Watches': ['Watches', 'watch'],
#     'perfume': ['perfume', 'deo', 'deodrant', 'cologne', 'EDT', 'body', 'notes', 'code', 'stone', 'men'],
#     'jewellery': ['jewellery', 'earrings', 'necklace', 'bangles', 'rings', 'plated', 'gold', 'set', 'necklace', 'pair'],
#     'lipstick': ['lipstick', 'lip color', 'lip crayon', 'color', 'it', 'lips', 'that', 'high'],
#     'foundation': ['foundation', 'concealer', 'face makeup', 'wardrobe', 'year', 'striped', 'regular', 'new'],
#     'nail polish': ['nail polish', 'nail paint', 'nail lacquer'],
#     'eyeliner': ['eyeliner', 'kajal', 'eye makeup', 'that', 'precision', 'this', 'it', 'ink'],
#     'wallet': ['wallet', 'card holder', 'card', 'fold', 'solid', 'zip', 'main'],
#     'sunglasses': ['sunglasses', 'shades', 'glares', 'lens', 'colour', 'feature', 'lensframe', 'material'],
#     'cap': ['cap', 'hat', 'black', 'solid', 'blue', 'design', 'set'],
#     'belt': ['belt', 'waist belt', 'rise', 'jeans', 'blue', 'mid', 'look'],
#     'socks': ['socks', 'anklets', 'length', 'pack', 'ankle', 'assorted', 'mouth']
# }
    
#     csv_path = "/home/artisans15/projects/fashion_retail_analytics/data/raw/myntra_products_catalog.csv"
#     if os.path.exists(csv_path):
#         try:
#             df = pd.read_csv(csv_path).head(20000)
            
#             # Calculate product popularity (simulate based on data patterns)
#             product_names = df['product_name'].value_counts()
#             for name, count in product_names.items():
#                 product_popularity[str(name).lower()] = count

#             for idx, row in df.iterrows():
#                 gender = str(row.get('Gender', 'Unisex')).strip()
#                 if gender.lower() in ['nan', 'unisex', '']:
#                     gender = 'Unisex'
                
#                 brand = str(row.get('ProductBrand', '')).strip()
#                 product_name = str(row.get('product_name', '')).strip()
#                 category = str(row.get('Category', '')).strip()
#                 price = row.get('Price', 0)
                
#                 # Track brand frequency
#                 if brand and len(brand) > 1:
#                     brand_frequency[brand.lower()] += 1
#                     brand_products[brand.lower()].append(product_name)
                
#                 # Track price ranges per category
#                 if price and price > 0:
#                     price_ranges[category.lower()].append(float(price))
                
#                 # Enhanced product name processing
#                 if product_name and len(product_name) > 1:
#                     key = product_name.lower()
#                     words[key] = {
#                         'context': {
#                             'type': 'product',
#                             'category': category,
#                             'valid_genders': [gender],
#                             'brand': brand,
#                             'price': price,
#                             'popularity': product_popularity.get(key, 0)
#                         }
#                     }
                    
#                     # Enhanced n-gram generation with context
#                     tokens = product_name.lower().split()
#                     for n in range(2, min(5, len(tokens)+1)):
#                         for i in range(len(tokens)-n+1):
#                             phrase = " ".join(tokens[i:i+n])
#                             if phrase not in words:
#                                 words[phrase] = {
#                                     'context': {
#                                         'type': 'phrase',
#                                         'category': category,
#                                         'valid_genders': [gender],
#                                         'brand': brand,
#                                         'source_product': product_name,
#                                         'popularity': product_popularity.get(product_name.lower(), 0)
#                                     }
#                                 }
                
#                 # Enhanced brand processing
#                 if brand and len(brand) > 1:
#                     brand_key = brand.lower()
#                     if brand_key not in words:
#                         words[brand_key] = {
#                             'context': {
#                                 'type': 'brand',
#                                 'category': category,
#                                 'valid_genders': [gender],
#                                 'product_count': brand_frequency[brand_key]
#                             }
#                         }
#                     else:
#                         if gender not in words[brand_key]['context']['valid_genders']:
#                             words[brand_key]['context']['valid_genders'].append(gender)
#                         words[brand_key]['context']['product_count'] = brand_frequency[brand_key]

#                 # Enhanced word-level processing
#                 product_words = product_name.split()
#                 for word in product_words:
#                     clean_word = re.sub(r'[^\w]', '', word.lower())
#                     if len(clean_word) > 2:
#                         # Smart category detection
#                         word_category = None
#                         for cat, valid_genders in category_rules.items():
#                             if cat in clean_word or clean_word in cat:
#                                 word_category = cat
#                                 break
                        
#                         if not word_category:
#                             word_category = category.lower() if category else 'general'
                        
#                         if clean_word not in words:
#                             words[clean_word] = {
#                                 'context': {
#                                     'type': 'keyword',
#                                     'category': word_category,
#                                     'valid_genders': [gender],
#                                     'frequency': 1
#                                 }
#                             }
#                         else:
#                             # Update frequency and genders - FIXED VERSION
#                             if 'frequency' not in words[clean_word]['context']:
#                                 words[clean_word]['context']['frequency'] = 1
#                             else:
#                                 words[clean_word]['context']['frequency'] += 1
                            
#                             if gender not in words[clean_word]['context']['valid_genders']:
#                                 words[clean_word]['context']['valid_genders'].append(gender)

                        
#                         # Store in enhanced product database
#                         product_database[word_category][gender].append({
#                             'name': product_name,
#                             'brand': brand,
#                             'word': clean_word,
#                             'price': price,
#                             'popularity': product_popularity.get(product_name.lower(), 0)
#                         })
                        
#                         category_gender_mapping[word_category].add(gender)
            
#             st.success(f"Enhanced index built with {len(df)} products, {len(brand_frequency)} brands.")
#         except Exception as e:
#             st.warning(f"Could not load CSV: {e}. Using enhanced sample data.")
#             # Enhanced sample data with more context
#             sample_data = {
#                 'saree': {'context': {'type': 'product', 'category': 'saree', 'valid_genders': ['Women'], 'popularity': 100}},
#                 'kurta': {'context': {'type': 'product', 'category': 'kurta', 'valid_genders': ['Men', 'Women'], 'popularity': 95}},
#                 'salawar': {'context': {'type': 'product', 'category': 'salwar', 'valid_genders': ['Women'], 'popularity': 90}},
#                 'shirt': {'context': {'type': 'product', 'category': 'shirt', 'valid_genders': ['Men', 'Women'], 'popularity': 90}},
#                 'dress': {'context': {'type': 'product', 'category': 'dress', 'valid_genders': ['Women'], 'popularity': 85}},
#                 'jeans': {'context': {'type': 'product', 'category': 'jeans', 'valid_genders': ['Men', 'Women'], 'popularity': 92}},
#                 'lehenga': {'context': {'type': 'product', 'category': 'lehenga', 'valid_genders': ['Women'], 'popularity': 75}},
#                 'nike': {'context': {'type': 'brand', 'category': 'sportswear', 'valid_genders': ['Men', 'Women'], 'product_count': 500}},
#                 'adidas': {'context': {'type': 'brand', 'category': 'sportswear', 'valid_genders': ['Men', 'Women'], 'product_count': 450}},
#                 'puma': {'context': {'type': 'brand', 'category': 'sportswear', 'valid_genders': ['Men', 'Women'], 'product_count': 400}},
#                 'shoes': {'context': {'type': 'product', 'category': 'shoes', 'valid_genders': ['Men', 'Women'], 'popularity': 80}},
#                 'sandals': {'context': {'type': 'product', 'category': 'sandals', 'valid_genders': ['Women'], 'popularity': 70}},
#                 'heels': {'context': {'type': 'product', 'category': 'heels', 'valid_genders': ['Women'], 'popularity': 60}},
#                 'watch': {'context': {'type': 'product', 'category': 'watch', 'valid_genders': ['Men', 'Women'], 'popularity': 90}},
#                 'handbag': {'context': {'type': 'product', 'category': 'handbag', 'valid_genders': ['Women'], 'popularity': 80}},
#                 'wallet': {'context': {'type': 'product', 'category': 'wallet', 'valid_genders': ['men'], 'popularity': 70}},
#                 't-shirt': {'context': {'type': 'product', 'category': 't-shirt', 'valid_genders': ['Men', 'Women'], 'popularity': 90}},
#                 'cap': {'context': {'type': 'product', 'category': 'cap', 'valid_genders': ['Men', 'Women'], 'popularity': 80}},
#                 'belt': {'context': {'type': 'product', 'category': 'belt', 'valid_genders': ['Men', 'Women'], 'popularity': 70}},
#                 'socks': {'context': {'type': 'product', 'category': 'socks', 'valid_genders': ['Men', 'Women'], 'popularity': 90}},
#                 'formal shoes': {'context': {'type': 'product', 'category': 'formal shoes', 'valid_genders': ['Men', 'Women'], 'popularity': 80}},
#                 'casual shoes': {'context': {'type': 'product', 'category': 'casual shoes', 'valid_genders': ['Men', 'Women'], 'popularity': 70}},
#                 'loafers': {'context': {'type': 'product', 'category': 'loafers', 'valid_genders': ['Men', 'Women'], 'popularity': 90}},
#                 'sunglasses': {'context': {'type': 'product', 'category': 'sunglasses', 'valid_genders': ['Men', 'Women'], 'popularity': 80}},
#                 'perfume': {'context': {'type': 'product', 'category': 'perfume', 'valid_genders': ['Men', 'Women'], 'popularity': 70}},
#                 'jacket': {'context': {'type': 'product', 'category': 'jacket', 'valid_genders': ['Men', 'Women'], 'popularity': 90}},
#                 'sweater': {'context': {'type': 'product', 'category': 'sweater', 'valid_genders': ['Men', 'Women'], 'popularity': 80}},
#                 'hoodie': {'context': {'type': 'product', 'category': 'hoodie', 'valid_genders': ['Men', 'Women'], 'popularity': 70}},
#                 'backpack': {'context': {'type': 'product', 'category': 'backpack', 'valid_genders': ['Men', 'Women'], 'popularity': 90}},
#                 'jewelry': {'context': {'type': 'product', 'category': 'jewelry', 'valid_genders': ['Men', 'Women'], 'popularity': 80}},
#                 'earrings': {'context': {'type': 'product', 'category': 'earrings', 'valid_genders': ['Men', 'Women'], 'popularity': 70}},
#                 'necklace': {'context': {'type': 'product', 'category': 'necklace', 'valid_genders': ['Women'], 'popularity': 90}},
#                 'bracelet': {'context': {'type': 'product', 'category': 'bracelet', 'valid_genders': ['Men', 'Women'], 'popularity': 80}},
#                 'lipstick': {'context': {'type': 'product', 'category': 'lipstick', 'valid_genders': ['Women'], 'popularity': 70}},
#                 'foundation': {'context': {'type': 'product', 'category': 'foundation', 'valid_genders': ['Women'], 'popularity': 90}},
#                 'nail polish': {'context': {'type': 'product', 'category': 'nail polish', 'valid_genders': ['Women'], 'popularity': 80}},
#                 'eyeliner': {'context': {'type': 'product', 'category': 'eyeliner', 'valid_genders': ['Women'], 'popularity': 70}},
#                 'nightwear': {'context': {'type': 'product', 'category': 'nightwear', 'valid_genders': ['Women'], 'popularity': 90}},
#                 'sportswear': {'context': {'type': 'product', 'category': 'sportswear', 'valid_genders': ['Men', 'Women'], 'popularity': 80}},
#                 'swimwear': {'context': {'type': 'product', 'category': 'swimwear', 'valid_genders': ['Women'], 'popularity': 70}},
#                 'earphones': {'context': {'type': 'product', 'category': 'earphones', 'valid_genders': ['Men', 'Women'], 'popularity': 90}}
#             }
#             words.update(sample_data)
    
#     # Create enhanced autocomplete instance
#     autocomplete_instance = AutoComplete(words=words)
#     return autocomplete_instance

def initialize_autocomplete():
    """Enhanced initialization focusing on product names from CSV"""
    global autocomplete_instance, product_database, category_gender_mapping
    global brand_frequency, product_popularity, category_keywords, brand_products, price_ranges
    
    if autocomplete_instance is not None:
        return autocomplete_instance
    
    words = {}
    
    csv_path = "/home/artisans15/projects/fashion_retail_analytics/data/raw/myntra_products_catalog.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path).head(20000)
            st.success(f"Loading {len(df)} products from CSV...")
            
            # Calculate product popularity (simulate based on data patterns)
            product_names = df['product_name'].value_counts()
            for name, count in product_names.items():
                product_popularity[str(name).lower()] = count

            for idx, row in df.iterrows():
                gender = str(row.get('Gender', 'Unisex')).strip()
                if gender.lower() in ['nan', 'unisex', '']:
                    gender = 'Unisex'
                
                brand = str(row.get('ProductBrand', '')).strip()
                product_name = str(row.get('product_name', '')).strip()
                category = str(row.get('Category', '')).strip()
                price = row.get('Price', 0)
                
                # Track brand frequency
                if brand and len(brand) > 1:
                    brand_frequency[brand.lower()] += 1
                    brand_products[brand.lower()].append(product_name)
                
                # Track price ranges per category
                if price and price > 0:
                    price_ranges[category.lower()].append(float(price))
                
                # **PRIMARY FOCUS: Product name processing for autocomplete**
                if product_name and len(product_name) > 1:
                    key = product_name.lower()
                    words[key] = {
                        'context': {
                            'type': 'product',
                            'category': category,
                            'valid_genders': [gender],
                            'brand': brand,
                            'price': price,
                            'popularity': product_popularity.get(key, 0)
                        }
                    }
                    
                    # Add product name tokens for partial matching
                    tokens = product_name.lower().split()
                    for token in tokens:
                        clean_token = re.sub(r'[^\w]', '', token)
                        if len(clean_token) > 2:  # Skip very short words
                            if clean_token not in words:
                                words[clean_token] = {
                                    'context': {
                                        'type': 'product_token',
                                        'category': category,
                                        'valid_genders': [gender],
                                        'brand': brand,
                                        'source_product': product_name,
                                        'popularity': product_popularity.get(key, 0)
                                    }
                                }
                            else:
                                # Update existing token context
                                if gender not in words[clean_token]['context']['valid_genders']:
                                    words[clean_token]['context']['valid_genders'].append(gender)
                    
                    # Add n-grams (2-4 words) from product names for better matching
                    for n in range(2, min(5, len(tokens) + 1)):
                        for i in range(len(tokens) - n + 1):
                            phrase = " ".join(tokens[i:i+n])
                            if phrase not in words and len(phrase) > 4:  # Skip very short phrases
                                words[phrase] = {
                                    'context': {
                                        'type': 'product_phrase',
                                        'category': category,
                                        'valid_genders': [gender],
                                        'brand': brand,
                                        'source_product': product_name,
                                        'popularity': product_popularity.get(key, 0)
                                    }
                                }
                
                # Store in product database for contextual suggestions
                if category:
                    product_database[category.lower()][gender].append({
                        'name': product_name,
                        'brand': brand,
                        'price': price,
                        'popularity': product_popularity.get(product_name.lower(), 0)
                    })
                    category_gender_mapping[category.lower()].add(gender)
            
            st.success(f"Enhanced index built with {len(df)} products, {len(words)} searchable terms.")
            
        except Exception as e:
            st.error(f"Could not load CSV: {e}")
            return None
    else:
        st.error(f"CSV file not found at: {csv_path}")
        return None
    
    # Create enhanced autocomplete instance
    autocomplete_instance = AutoComplete(words=words)
    return autocomplete_instance


# Load product names for enhanced fuzzy matching
all_product_names = []
product_metadata = {}

if os.path.exists("/home/artisans15/projects/fashion_retail_analytics/data/raw/myntra_products_catalog.csv"):
    try:
        df = pd.read_csv("/home/artisans15/projects/fashion_retail_analytics/data/raw/myntra_products_catalog.csv").head(20000)
        all_product_names = df['product_name'].dropna().astype(str).str.lower().unique().tolist()
        
        # Build metadata for products
        for _, row in df.iterrows():
            name = str(row.get('product_name', '')).lower()
            if name:
                product_metadata[name] = {
                    'brand': str(row.get('ProductBrand', '')),
                    'category': str(row.get('Category', '')),
                    'gender': str(row.get('Gender', 'Unisex')),
                    'price': row.get('Price', 0)
                }
    except Exception as e:
        st.warning(f"Could not load product metadata: {e}")

# def get_smart_suggestions(query: str, limit: int = 10):
#     """Amazon/Myntra-level intelligent suggestion generation"""
#     if not query or len(query) < 1:
#         return []

#     # Preprocess query
#     processed_query = preprocess_query(query)
#     intents = extract_intent(query)
    
#     autocomplete = initialize_autocomplete()
#     suggestions_with_scores = []
#     seen = set()

#     # Track search pattern for learning
#     search_patterns[processed_query] += 1

#     # --- 1. Enhanced exact and prefix matches ---
#     exact_matches = []
#     for word in autocomplete.words:
#         if word == processed_query:
#             context = autocomplete.words[word]['context']
#             score = calculate_relevance_score(word, processed_query, context)
#             exact_matches.append((score + 200, word, context))  # Highest priority
    
#     exact_matches.sort(reverse=True)
#     for score, word, context in exact_matches[:3]:
#         if word not in seen:
#             suggestions_with_scores.append((score, {
#                 "suggestion": word,
#                 "type": "Exact Match",
#                 "scope": f"for {'/'.join(context.get('valid_genders', ['Everyone']))}",
#                 "category": context.get('category', 'general'),
#                 "gender": context.get('valid_genders', ['Unisex'])[0],
#                 "_score": score
#             }))
#             seen.add(word)

#     # --- 2. Enhanced fuzzy product matching ---
#     if all_product_names:
#         fuzzy_matches = process.extract(
#             processed_query, 
#             all_product_names, 
#             scorer=fuzz.token_sort_ratio, 
#             limit=limit * 4
#         )
        
#         for match, fuzzy_score, _ in fuzzy_matches:
#             if fuzzy_score > 70 and match not in seen:
#                 metadata = product_metadata.get(match, {})
                
#                 # Apply intent filtering
#                 if 'gender' in intents:
#                     query_gender = 'Women' if intents['gender'] in ['women', 'womens', 'female'] else 'Men'
#                     if metadata.get('gender', 'Unisex') not in [query_gender, 'Unisex']:
#                         continue
                
#                 final_score = calculate_relevance_score(match, processed_query, {'type': 'product', **metadata})
#                 suggestions_with_scores.append((final_score + fuzzy_score, {
#                     "suggestion": match.title(),
#                     "type": "Product Match",
#                     "scope": f"{metadata.get('brand', 'Brand')} • {metadata.get('gender', 'Unisex')}",
#                     "category": metadata.get('category', 'general'),
#                     "gender": metadata.get('gender', 'Unisex'),
#                     "_score": final_score + fuzzy_score
#                 }))
#                 seen.add(match)

#     # --- 3. Enhanced autocomplete with better scoring ---
#     try:
#         raw_suggestions = autocomplete.search(word=processed_query, max_cost=3, size=limit * 5)
#         suggestions_flat = [item for sublist in raw_suggestions for item in sublist]

#         for sug in suggestions_flat:
#             sug_lower = sug.lower()
#             if sug_lower in seen or len(sug) < 2:
#                 continue

#             context_info = autocomplete.words.get(sug_lower, {}).get('context', {})
            
#             # Advanced relevance calculation
#             relevance_score = calculate_relevance_score(sug, processed_query, context_info)
            
#             # Apply intent-based filtering and boosting
#             valid_genders = context_info.get('valid_genders', ['Unisex'])
            
#             # Gender filtering based on intent
#             if 'gender' in intents:
#                 query_gender = 'Women' if intents['gender'] in ['women', 'womens', 'female', 'girls'] else 'Men'
#                 if query_gender not in valid_genders and 'Unisex' not in valid_genders:
#                     continue
#                 else:
#                     relevance_score += 20  # Boost gender-relevant results

#             category = context_info.get('category', 'general')
#             product_type = context_info.get('type', 'product')
            
#             # Create suggestion for each valid gender (Amazon-style)
#             for gender in valid_genders:
#                 if len(suggestions_with_scores) >= limit * 3:
#                     break
                    
#                 scope_text = f"for {gender}" if gender != 'Unisex' else "for Everyone"
#                 if 'brand' in intents or product_type == 'brand':
#                     scope_text = f"Brand • {scope_text}"
                
#                 suggestions_with_scores.append((relevance_score, {
#                     "suggestion": sug.title(),
#                     "type": product_type.capitalize(),
#                     "scope": scope_text,
#                     "category": category,
#                     "gender": gender,
#                     "_score": relevance_score
#                 }))
            
#             seen.add(sug_lower)

#     except Exception as e:
#         st.error(f"Autocomplete error: {e}")

#     # --- 4. Add contextual suggestions ---
#     contextual_suggestions = get_contextual_suggestions(
#         processed_query, 
#         category_hint=intents.get('category'),
#         gender_hint=intents.get('gender')
#     )
    
#     for ctx_sug in contextual_suggestions:
#         if ctx_sug['suggestion'].lower() not in seen:
#             suggestions_with_scores.append((ctx_sug['_score'], ctx_sug))
#             seen.add(ctx_sug['suggestion'].lower())

#     # --- 5. Enhanced sorting and grouping ---
#     suggestions_with_scores.sort(key=lambda x: x[0], reverse=True)
#     raw_suggestions = [s[1] for s in suggestions_with_scores]
    
#     # Apply advanced grouping and deduplication
#     final_suggestions = group_and_deduplicate_suggestions(raw_suggestions, processed_query)
    
#     # --- 6. Post-processing for better UX ---
#     for i, suggestion in enumerate(final_suggestions):
#         # Add query completion hints
#         original_suggestion = suggestion['suggestion']
        
#         # Highlight completion part (Amazon-style)
#         if original_suggestion.lower().startswith(query.lower()):
#             completion_part = original_suggestion[len(query):]
#             if completion_part:
#                 suggestion['completion_hint'] = completion_part
        
#         # Add trending indicators for popular items
#         if suggestion.get('_score', 0) > 150:
#             suggestion['trending'] = True
    
#     return final_suggestions[:limit]

def get_smart_suggestions(query: str, limit: int = 10):
    """Product name focused suggestion generation"""
    if not query or len(query) < 1:
        return []

    # Preprocess query
    processed_query = preprocess_query(query)
    intents = extract_intent(query)
    
    autocomplete = initialize_autocomplete()
    if not autocomplete:
        return []
    
    suggestions_with_scores = []
    seen = set()

    # --- 1. Direct product name matches (HIGHEST PRIORITY) ---
    if all_product_names:
        # Exact matches first
        exact_product_matches = [name for name in all_product_names if name.lower() == processed_query.lower()]
        for match in exact_product_matches[:3]:
            if match not in seen:
                metadata = product_metadata.get(match, {})
                suggestions_with_scores.append((300, {  # Highest score
                    "suggestion": match.title(),
                    "type": "Exact Product",
                    "scope": f"{metadata.get('brand', 'Brand')} • {metadata.get('gender', 'Unisex')}",
                    "category": metadata.get('category', 'general'),
                    "gender": metadata.get('gender', 'Unisex'),
                    "_score": 300
                }))
                seen.add(match)
        
        # Fuzzy product name matches
        fuzzy_matches = process.extract(
            processed_query, 
            all_product_names, 
            scorer=fuzz.token_sort_ratio, 
            limit=limit * 3
        )
        
        for match, fuzzy_score, _ in fuzzy_matches:
            if fuzzy_score > 60 and match not in seen:  # Lower threshold for more results
                metadata = product_metadata.get(match, {})
                
                # Apply gender filtering if specified in query
                if 'gender' in intents:
                    query_gender = 'Women' if intents['gender'] in ['women', 'womens', 'female'] else 'Men'
                    if metadata.get('gender', 'Unisex') not in [query_gender, 'Unisex']:
                        continue
                
                final_score = fuzzy_score + 100  # Boost product names
                suggestions_with_scores.append((final_score, {
                    "suggestion": match.title(),
                    "type": "Product",
                    "scope": f"{metadata.get('brand', 'Brand')} • {metadata.get('gender', 'Unisex')}",
                    "category": metadata.get('category', 'general'),
                    "gender": metadata.get('gender', 'Unisex'),
                    "_score": final_score
                }))
                seen.add(match)

    # --- 2. Autocomplete from indexed words (tokens and phrases) ---
    try:
        raw_suggestions = autocomplete.search(word=processed_query, max_cost=3, size=limit * 5)
        suggestions_flat = [item for sublist in raw_suggestions for item in sublist]

        for sug in suggestions_flat:
            sug_lower = sug.lower()
            if sug_lower in seen or len(sug) < 2:
                continue

            context_info = autocomplete.words.get(sug_lower, {}).get('context', {})
            
            # Prioritize actual products over tokens
            if context_info.get('type') == 'product':
                relevance_score = 200
            elif context_info.get('type') == 'product_phrase':
                relevance_score = 150
            else:
                relevance_score = 100
                
            # Calculate additional relevance
            relevance_score += calculate_relevance_score(sug, processed_query, context_info)
            
            # Apply gender filtering
            valid_genders = context_info.get('valid_genders', ['Unisex'])
            if 'gender' in intents:
                query_gender = 'Women' if intents['gender'] in ['women', 'womens', 'female', 'girls'] else 'Men'
                if query_gender not in valid_genders and 'Unisex' not in valid_genders:
                    continue
                else:
                    relevance_score += 20

            # Create suggestion
            suggestions_with_scores.append((relevance_score, {
                "suggestion": sug.title(),
                "type": context_info.get('type', 'product').replace('_', ' ').title(),
                "scope": f"for {'/'.join(valid_genders)}",
                "category": context_info.get('category', 'general'),
                "gender": valid_genders[0] if valid_genders else 'Unisex',
                "_score": relevance_score
            }))
            seen.add(sug_lower)

    except Exception as e:
        st.error(f"Autocomplete error: {e}")

    # --- 3. Sort and return top suggestions ---
    suggestions_with_scores.sort(key=lambda x: x[0], reverse=True)
    final_suggestions = [s[1] for s in suggestions_with_scores[:limit]]
    
    return final_suggestions


@api_app.get("/suggestions")
async def suggestions_endpoint(q: str = Query(..., min_length=1), limit: int = 10):
    """API endpoint for getting structured search suggestions"""
    return get_smart_suggestions(q, limit)

@api_app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "autocomplete_loaded": autocomplete_instance is not None,
        "categories": len(category_gender_mapping),
        "products": sum(len(genders) for genders in product_database.values())
    }

def run_api():
    uvicorn.run(api_app, host="127.0.0.1", port=8001, log_level="error")

# Start API server
if 'api_thread_started' not in st.session_state:
    with st.spinner("Loading gender-category aware autocomplete engine..."):
        initialize_autocomplete()
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    st.session_state.api_thread_started = True

# --- Enhanced Streamlit UI ---
st.title("🛍️ Product Search with Smart Autocomplete")
st.markdown("*Smart suggestions for better product discovery*")

if autocomplete_instance:
    st.sidebar.success(f"**Enhanced Search Engine Ready!**")
    st.sidebar.info(f"**Indexed Terms:** {len(autocomplete_instance.words):,}")
    st.sidebar.info(f"**Product Categories:** {len(category_gender_mapping)}")
    st.sidebar.markdown("### Smart Features:")
    st.sidebar.markdown("- 👥 Gender-Aware Suggestions")
    st.sidebar.markdown("- 🏷️ Category-Based Filtering")
    st.sidebar.markdown("- 🚫 Invalid Combination Prevention")
    st.sidebar.markdown("- ✨ Context-Aware Results")

# Enhanced search interface with better styling
search_html = r"""
<div style="position: relative; display: inline-block; width: 100%;">
    <input type="text" id="searchInput" placeholder="Search products (try: saree, kurta, shirt...)" 
           style="width:100%; max-width:500px; font-size:18px; padding:15px; border: 3px solid #FF9900; 
                  border-radius: 12px; outline: none; box-shadow: 0 2px 8px rgba(255,153,0,0.3);"/>
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
        overflow: hidden;
    ">
        <div style="display: flex; width: 100%; height: 100%;">
            <div id="suggestions-list" style="width: 60%; border-right: 1px solid #eee; overflow-y: auto;"></div>
            <div id="suggestion-details" style="width: 40%; padding: 20px; background-color: #f8f8f8;">
                <p style="color: #777; text-align: center; margin-top: 20px;">Hover over a suggestion to see details.</p>
            </div>
        </div>
    </div>
    <div id="loading" style="display: none; padding: 10px; text-align: center; color: #666;">
        🔍 Searching...
    </div>
</div>
<style>
.suggestion-item {
    padding: 12px 15px;
    cursor: pointer;
    border-bottom: 1px solid #f5f5f5;
    transition: background-color 0.2s ease;
    font-size: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.suggestion-item:hover, .suggestion-item.selected {
    background-color: #f0f8ff;
}
.suggestion-text {
    color: #111;
}
.suggestion-scope {
    font-size: 14px;
    color: #555;
    margin-left: 8px;
    white-space: nowrap;
    background: #e8f4f8;
    padding: 2px 8px;
    border-radius: 12px;
}
.suggestion-scope.women {
    background: #ffe8f0;
    color: #d63384;
}
.suggestion-scope.men {
    background: #e8f0ff;
    color: #0066cc;
}
.suggestion-text strong {
    font-weight: 600;
    color: #000;
}
</style>
<script>
const searchInput = document.getElementById("searchInput");
const suggestionsContainer = document.getElementById("suggestions-container");
const suggestionsList = document.getElementById("suggestions-list");
const suggestionDetails = document.getElementById("suggestion-details");
const loadingDiv = document.getElementById("loading");
let searchTimeout;
let selectedIndex = -1;

searchInput.addEventListener("input", function() {
    clearTimeout(searchTimeout);
    let query = this.value.trim();
    if (query.length < 1) {
        hideSuggestions();
        return;
    }
    loadingDiv.style.display = 'block';
    searchTimeout = setTimeout(() => fetchSuggestions(query), 250);
});

function fetchSuggestions(query) {
    fetch(`http://127.0.0.1:8001/suggestions?q=${encodeURIComponent(query)}&limit=8`)
        .then(response => response.json())
        .then(data => {
            loadingDiv.style.display = 'none';
            suggestionsList.innerHTML = "";
            
            if (data.length > 0) {
                data.forEach((item, index) => {
                    let div = document.createElement('div');
                    div.className = 'suggestion-item';
                    
                    let highlightedText = highlightCompletion(item.suggestion, query);
                    
                    // Add gender-specific styling
                    let scopeClass = '';
                    if (item.scope && item.scope.toLowerCase().includes('women')) {
                        scopeClass = 'women';
                    } else if (item.scope && item.scope.toLowerCase().includes('men')) {
                        scopeClass = 'men';
                    }
                    
                    div.innerHTML = `
                        <span class="suggestion-text">${highlightedText}</span>
                        <span class="suggestion-scope ${scopeClass}">${item.scope}</span>
                    `;
                    
                    div.dataset.suggestion = item.suggestion;
                    div.dataset.type = item.type;
                    div.dataset.scope = item.scope;
                    div.dataset.category = item.category || 'general';
                    div.dataset.gender = item.gender || 'Unisex';

                    div.addEventListener('mouseover', () => {
                        selectedIndex = index;
                        updateSelection(suggestionsList.children);
                        updateDetails(item);
                    });

                    div.onclick = () => {
                        searchInput.value = item.suggestion;
                        hideSuggestions();
                    };
                    
                    suggestionsList.appendChild(div);
                });
                showSuggestions();
            } else {
                hideSuggestions();
            }
        })
        .catch(error => {
            console.error('Search error:', error);
            hideSuggestions();
        });
}

function highlightCompletion(text, query) {
    const lowerText = text.toLowerCase();
    const lowerQuery = query.toLowerCase();
    const startIndex = lowerText.indexOf(lowerQuery);

    if (startIndex === 0) { 
        const unchangedPart = text.substring(0, query.length);
        const boldPart = text.substring(query.length);
        return `${unchangedPart}<strong>${boldPart}</strong>`;
    }
    return text;
}

function updateDetails(item) {
    const iconMap = {
        'Brand': '🏢', 'Product': '👟', 'Category': '🏷️', 'Contextual': '🎯', 'Query': '🔍'
    };
    const icon = iconMap[item.type] || '🛍️';

    // Gender-specific emojis
    let genderIcon = '👤';
    if (item.gender === 'Women') genderIcon = '👩';
    else if (item.gender === 'Men') genderIcon = '👨';

    suggestionDetails.innerHTML = `
        <div style="text-align: center; font-size: 48px; margin-bottom: 10px;">${icon}</div>
        <h3 style="margin: 0; text-align: center;">${item.suggestion}</h3>
        <p style="color: #555; text-align: center;">Type: ${item.type}</p>
        <p style="color: #555; text-align: center;">${genderIcon} ${item.scope || 'General'}</p>
        ${item.category ? `<p style="color: #888; text-align: center; font-size: 14px;">Category: ${item.category}</p>` : ''}
        <div style="background: #f0f8ff; padding: 10px; border-radius: 8px; margin-top: 15px;">
            <p style="font-size:12px; color: #666; text-align: center; margin: 0;">
                ✅ Gender-appropriate suggestion<br>
                Based on product category data
            </p>
        </div>
    `;
}

function showSuggestions() {
    if (suggestionsList.children.length > 0) {
        suggestionsContainer.style.display = 'block';
    }
}

function hideSuggestions() {
    suggestionsContainer.style.display = 'none';
    loadingDiv.style.display = 'none';
    selectedIndex = -1;
}

// Keyboard navigation
searchInput.addEventListener('keydown', function(e) {
    const items = suggestionsList.children;
    if (items.length === 0) return;

    if (e.key === 'ArrowDown') {
        e.preventDefault();
        selectedIndex = (selectedIndex + 1) % items.length;
        updateSelection(items);
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        selectedIndex = (selectedIndex - 1 + items.length) % items.length;
        updateSelection(items);
    } else if (e.key === 'Enter') {
        e.preventDefault();
        if (selectedIndex > -1 && items[selectedIndex]) {
            searchInput.value = items[selectedIndex].dataset.suggestion;
            hideSuggestions();
        }
    } else if (e.key === 'Escape') {
        hideSuggestions();
    }
});

function updateSelection(items) {
    for (let i = 0; i < items.length; i++) {
        if (i === selectedIndex) {
            items[i].classList.add('selected');
            const selectedItem = items[i];
            updateDetails({
                suggestion: selectedItem.dataset.suggestion,
                type: selectedItem.dataset.type,
                scope: selectedItem.dataset.scope,
                category: selectedItem.dataset.category,
                gender: selectedItem.dataset.gender
            });
            selectedItem.scrollIntoView({ block: 'nearest' });
        } else {
            items[i].classList.remove('selected');
        }
    }
}

// Click outside to close
document.addEventListener('click', function(event) {
    if (!searchInput.contains(event.target) && !suggestionsContainer.contains(event.target)) {
        hideSuggestions();
    }
});

searchInput.addEventListener('focus', function() {
    if (this.value.length >= 1) {
        fetchSuggestions(this.value);
    }
});
</script>
"""

components.html(search_html, height=550)

# Display statistics
if autocomplete_instance:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Categories", len(category_gender_mapping))
    
    with col2:
        total_products = sum(len(genders) for genders in product_database.values())
        st.metric("Product Variations", total_products)
    
    with col3:
        st.metric("Indexed Words", len(autocomplete_instance.words))

    # Show category breakdown
    if st.expander("📊 Category-Gender Breakdown", expanded=False):
        for category, genders in list(category_gender_mapping.items())[:10]:
            st.write(f"**{category.capitalize()}:** {', '.join(genders)}")

# run the fastapi app with uvicorn
# uvicorn.run(api_app, host="127.0.0.1", port=8001)