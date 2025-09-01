# FastAPI Backend Server for E-commerce Search
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
from fast_autocomplete import AutoComplete
from rapidfuzz import fuzz, process
import os
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

# Initialize FastAPI app
app = FastAPI(title="E-commerce Search API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*",
                   "https://672797d3a000.ngrok-free.app",
                   "https://5d49d13071da.ngrok-free.app",
                   ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
autocomplete_instance = None
product_database = defaultdict(lambda: defaultdict(list))  # {category: {gender: [products]}}
category_gender_mapping = defaultdict(set)  # {category: {valid_genders}}
all_product_names = []

def initialize_autocomplete():
    """Initialize Fast-Autocomplete with enhanced product categorization"""
    global autocomplete_instance, product_database, category_gender_mapping, all_product_names
    
    if autocomplete_instance is not None:
        return autocomplete_instance
    
    # Base words with proper categorization
    words = {}
    
    # Gender-specific categories mapping
    category_rules = {
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
    }
    
    csv_path = "/home/artisans15/projects/fashion_retail_analytics/data/raw/myntra_products_catalog.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path).head(20000)
            all_product_names = df['product_name'].dropna().astype(str).str.lower().unique().tolist()

            for _, row in df.iterrows():
                gender = str(row.get('Gender', 'Unisex')).strip()
                if gender.lower() in ['nan', 'unisex', '']:
                    gender = 'Unisex'
                brand = str(row.get('ProductBrand', '')).strip()
                product_name = str(row.get('product_name', '')).strip()
                category = str(row.get('Category', '')).strip()
                
                # Only add to autocomplete if present in dataset
                if product_name and len(product_name) > 1:
                    key = product_name.lower()
                    words[key] = {
                        'context': {
                            'type': 'product',
                            'category': category,
                            'valid_genders': [gender],
                            'brand': brand
                        }
                    }
                
                # Add n-grams (phrases) from product name for better phrase search
                tokens = product_name.lower().split()
                for n in range(2, min(5, len(tokens)+1)):  # n-grams from 2 to 4 words
                    for i in range(len(tokens)-n+1):
                        phrase = " ".join(tokens[i:i+n])
                        if phrase not in words:
                            words[phrase] = {
                                'context': {
                                    'type': 'phrase',
                                    'category': category,
                                    'valid_genders': [gender],
                                    'brand': brand
                                }
                            }

                if brand and len(brand) > 1:
                    brand_key = brand.lower()
                    if brand_key not in words:
                        words[brand_key] = {
                            'context': {
                                'type': 'brand',
                                'category': category,
                                'valid_genders': [gender]
                            }
                        }
                    else:
                        if gender not in words[brand_key]['context']['valid_genders']:
                            words[brand_key]['context']['valid_genders'].append(gender)
                
                # Process product name and categorize
                product_words = product_name.split()
                for word in product_words:
                    clean_word = word.strip('.,!?-')
                    if len(clean_word) > 2:
                        # Determine category from word
                        word_category = None
                        for cat, valid_genders in category_rules.items():
                            if cat in clean_word:
                                word_category = cat
                                break
                        
                        if not word_category:
                            word_category = category.lower() if category else 'general'
                        
                        if clean_word not in words:
                            words[clean_word] = {
                                'context': {
                                    'type': 'product',
                                    'category': word_category,
                                    'valid_genders': [gender]
                                }
                            }
                        else:
                            # Add gender to existing product
                            if gender not in words[clean_word]['context']['valid_genders']:
                                words[clean_word]['context']['valid_genders'].append(gender)
                        
                        # Store in product database
                        product_database[word_category][gender].append({
                            'name': product_name,
                            'brand': brand,
                            'word': clean_word
                        })
                        
                        # Update category-gender mapping
                        category_gender_mapping[word_category].add(gender)
            
            print(f"Loaded {len(df)} products with gender-category mapping.")
        except Exception as e:
            print(f"Could not load CSV: {e}. Using sample data.")
            # Add sample data with proper categorization
            sample_data = {
                'saree': {'context': {'type': 'product', 'category': 'saree', 'valid_genders': ['Women']}},
                'kurta': {'context': {'type': 'product', 'category': 'kurta', 'valid_genders': ['Men', 'Women']}},
                'shirt': {'context': {'type': 'product', 'category': 'shirt', 'valid_genders': ['Men', 'Women']}},
                'dress': {'context': {'type': 'product', 'category': 'dress', 'valid_genders': ['Women']}},
                'jeans': {'context': {'type': 'product', 'category': 'jeans', 'valid_genders': ['Men', 'Women']}},
                'lehenga': {'context': {'type': 'product', 'category': 'lehenga', 'valid_genders': ['Women']}},
                'shoes': {'context': {'type': 'product', 'category': 'shoes', 'valid_genders': ['Men', 'Women']}},
                'sandals': {'context': {'type': 'product', 'category': 'sandals', 'valid_genders': ['Men', 'Women']}},
                'heels': {'context': {'type': 'product', 'category': 'heels', 'valid_genders': ['Women']}},
                'watch': {'context': {'type': 'product', 'category': 'watch', 'valid_genders': ['Men', 'Women']}},
                'handbag': {'context': {'type': 'product', 'category': 'handbag', 'valid_genders': ['Women']}},
                'wallet': {'context': {'type': 'product', 'category': 'wallet', 'valid_genders': ['Men', 'Women']}},
                't-shirt': {'context': {'type': 'product', 'category': 't-shirt', 'valid_genders': ['Men', 'Women']}},
                'jacket': {'context': {'type': 'product', 'category': 'jackets', 'valid_genders': ['Men', 'Women']}},
                'skirt': {'context': {'type': 'product', 'category': 'skirt', 'valid_genders': ['Women']}},
                'blouse': {'context': {'type': 'product', 'category': 'blouse', 'valid_genders': ['Women']}},
                'backpack': {'context': {'type': 'product', 'category': 'backpack', 'valid_genders': ['Men', 'Women']}},
                'necklace': {'context': {'type': 'product', 'category': 'necklace', 'valid_genders': ['Women']}},
                'earrings': {'context': {'type': 'product', 'category': 'earrings', 'valid_genders': ['Women']}},
                'belt': {'context': {'type': 'product', 'category': 'belt', 'valid_genders': ['Men', 'Women']}},
                'cap': {'context': {'type': 'product', 'category': 'cap', 'valid_genders': ['Men', 'Women']}},
                'gloves': {'context': {'type': 'product', 'category': 'gloves', 'valid_genders': ['Men', 'Women']}},
                'scarf': {'context': {'type': 'product', 'category': 'scarf', 'valid_genders': ['Men', 'Women']}},
                'socks': {'context': {'type': 'product', 'category': 'socks', 'valid_genders': ['Men', 'Women']}},
                'tie': {'context': {'type': 'product', 'category': 'tie', 'valid_genders': ['Men']}},
                'cufflinks': {'context': {'type': 'product', 'category': 'cufflinks', 'valid_genders': ['Men']}},
                'lingerie': {'context': {'type': 'product', 'category': 'lingerie', 'valid_genders': ['Women']}},
                'nightwear': {'context': {'type': 'product', 'category': 'nightwear', 'valid_genders': ['Men', 'Women']}},
                'sportswear': {'context': {'type': 'product', 'category': 'sportswear', 'valid_genders': ['Men', 'Women']}},
                'swimwear': {'context': {'type': 'product', 'category': 'swimwear', 'valid_genders': ['Men', 'Women']}},
                'formal shoes': {'context': {'type': 'product', 'category': 'formal shoes', 'valid_genders': ['Men']}},
                'casual shoes': {'context': {'type': 'product', 'category': 'casual shoes', 'valid_genders': ['Men', 'Women']}},
                'loafers': {'context': {'type': 'product', 'category': 'loafers', 'valid_genders': ['Men', 'Women']}},
                'earphones': {'context': {'type': 'product', 'category': 'earphones', 'valid_genders': ['Men', 'Women']}},
                'sunglasses': {'context': {'type': 'product', 'category': 'sunglasses', 'valid_genders': ['Men', 'Women']}},
                'perfume': {'context': {'type': 'product', 'category': 'perfume', 'valid_genders': ['Men', 'Women']}},
                'hoodie': {'context': {'type': 'product', 'category': 'hoodie', 'valid_genders': ['Men', 'Women']}},
                'sweater': {'context': {'type': 'product', 'category': 'sweater', 'valid_genders': ['Men', 'Women']}},
            }
            words.update(sample_data)
    
    # Create autocomplete instance
    autocomplete_instance = AutoComplete(words=words)
    return autocomplete_instance

def get_smart_suggestions(query: str, limit: int = 10):
    """Get intelligent search suggestions with gender-category awareness"""
    if not query or len(query) < 1:
        return []

    autocomplete = initialize_autocomplete()
    suggestions_ranked = []
    seen = set()

    # --- 1. Full fuzzy matches ---
    if all_product_names:
        matches = process.extract(query, all_product_names, scorer=fuzz.token_sort_ratio, limit=limit * 3)
        for match, score, _ in matches:
            if score > 80 and match.lower() not in seen:
                suggestions_ranked.append((
                    score + 100,
                    {
                        "suggestion": match,
                        "type": "Product",
                        "scope": "from Catalog",
                        "category": "full-product",
                        "gender": "Unisex"
                    }
                ))
                seen.add(match.lower())

    # --- 2. Phrase-based autocomplete ---
    raw_suggestions = autocomplete.search(word=query, max_cost=2, size=limit * 3)
    suggestions_flat = [item for sublist in raw_suggestions for item in sublist]

    for sug in suggestions_flat:
        sug_lower = sug.lower()
        if sug_lower in seen or sug_lower == query.lower():
            continue

        context_info = None
        # Try to get context for the whole phrase first
        if sug_lower in autocomplete.words:
            context_info = autocomplete.words[sug_lower]['context']
        else:
            # Fallback: check for context in any word in the phrase
            for word in sug_lower.split():
                if word in autocomplete.words:
                    context_info = autocomplete.words[word]['context']
                    break
        if not context_info:
            continue

        category = context_info.get('category', 'general')
        valid_genders = context_info.get('valid_genders', ['Unisex'])
        product_type = context_info.get('type', 'product')

        for gender in valid_genders:
            suggestions_ranked.append((
                70 if product_type == 'phrase' else 50,
                {
                    "suggestion": sug,
                    "type": product_type.capitalize(),
                    "scope": f"for {gender}" if gender != 'Unisex' else "for Everyone",
                    "category": category,
                    "gender": gender
                }
            ))
        seen.add(sug_lower)

    # --- 3. Sort by score ---
    suggestions_ranked.sort(key=lambda x: x[0], reverse=True)
    final_suggestions = [s[1] for s in suggestions_ranked[:limit]]

    return final_suggestions

@app.get("/suggestions")
async def suggestions_endpoint(q: str = Query(..., min_length=1), limit: int = 10):
    """API endpoint for getting structured search suggestions"""
    return get_smart_suggestions(q, limit)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "autocomplete_loaded": autocomplete_instance is not None,
        "categories": len(category_gender_mapping),
        "products": sum(len(genders) for genders in product_database.values())
    }

@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    return {
        "total_categories": len(category_gender_mapping),
        "total_products": sum(len(genders) for genders in product_database.values()),
        "indexed_words": len(autocomplete_instance.words) if autocomplete_instance else 0,
        "category_breakdown": dict(list(category_gender_mapping.items())[:10])
    }

# Initialize autocomplete on startup
@app.on_event("startup")
async def startup_event():
    initialize_autocomplete()
    print("🚀 FastAPI server started with autocomplete engine loaded!")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010, log_level="info")