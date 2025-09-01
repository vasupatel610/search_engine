# app.py ── Enhanced Streamlit UI for the FastAPI search service

import streamlit as st
import requests
import re
from collections import Counter

# -------------------------------------------------------------------
# 1. Configuration & Custom CSS
# -------------------------------------------------------------------

API_URL = "http://localhost:8000"
HEALTH_EP = f"{API_URL}/health"
SEARCH_EP = f"{API_URL}/search"

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding: 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Remove default streamlit spacing */
    .block-container {
        padding: 1rem;
        max-width: 100%;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        color: #f0f0f0;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Search section styling */
    .search-section {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #e6e6e6;
    }
    
    /* Remove default streamlit element spacing */
    .element-container {
        margin: 0 !important;
    }
    
    /* Fix empty space issues */
    .stMarkdown {
        margin-bottom: 0 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
        width: 100%;
        margin-bottom: 0.5rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Product card styling - Updated */
    .product-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .product-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .product-title {
        color: #2c3e50;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        line-height: 1.4;
    }
    
    .price-tag {
        background: linear-gradient(45deg, #27ae60, #2ecc71);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        box-shadow: 0 2px 10px rgba(39, 174, 96, 0.3);
        text-align: center;
        min-width: 120px;
    }
    
    .description {
        color: #34495e;
        line-height: 1.6;
        margin-top: 1rem;
    }
    
    .details-badge {
        background: #ecf0f1;
        color: #2c3e50;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        display: inline-block;
        font-weight: 600;
    }
    
    /* Highlighting */
    mark {
        background: linear-gradient(120deg, #a8edea 0%, #fed6e3 100%);
        padding: 0.1rem 0.3rem;
        border-radius: 3px;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #f8f9fa;
    }
    
    /* Status indicators */
    .status-healthy {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 10px;
        font-weight: 600;
    }
    
    .status-error {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Search input styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e6e6e6;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e6e6e6;
        padding: 0.5rem;
        transition: all 0.3s ease;
    }
    
    /* Results counter */
    .results-counter {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Loading spinner customization */
    .stSpinner > div {
        border-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for the query
if 'query' not in st.session_state:
    st.session_state.query = ""

# Define the predefined questions with emojis
PREDEFINED_QUESTIONS = [
    "🎒 DKNY Unisex Trolley Bag",
    "👔 Navy Blue Casual Shirt Under 1500",
    "👗 Women Blue Printed Kurta",
    "👞 Men Formal Slip-Ons",
    "👖 Cropped Jeans for Women"
]

# -------------------------------------------------------------------
# 2. Check back-end status once at start-up
# -------------------------------------------------------------------

@st.cache_data(ttl=60)   # refresh every 60 s
def get_backend_status():
    """Checks the health of the FastAPI back-end."""
    try:
        return requests.get(HEALTH_EP, timeout=5).json()
    except Exception as e:
        return {"status": "unreachable", "detail": str(e)}

# -------------------------------------------------------------------
# 3. Header and Status Check
# -------------------------------------------------------------------

# Main header
st.markdown("""
<div class="main-header">
    <h1>🔍 Smart Product Search</h1>
    <p>Find exactly what you're looking for with natural language search</p>
</div>
""", unsafe_allow_html=True)

# Check backend status
status = get_backend_status()

# Sidebar for status
with st.sidebar:
    st.markdown("### 🔧 System Status")
    
    if status.get("status") == "healthy":
        st.markdown('<div class="status-healthy">✅ Backend Online</div>', unsafe_allow_html=True)
        st.success("All systems operational!")
    else:
        st.markdown('<div class="status-error">❌ Backend Offline</div>', unsafe_allow_html=True)
        st.error("Please start the FastAPI backend service")
        st.stop()
    
    st.markdown("### ℹ️ How to Use")
    st.info("""
    1. **Type naturally**: "blue jeans under 2000"
    2. **Use quick options**: Click suggested searches
    3. **Adjust results**: Set how many to show
    4. **Smart ranking**: Results auto-sorted by relevance
    """)
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Include **colors** (blue, red, black)
    - Mention **price range** (under 1500)
    - Specify **category** (shirt, jeans, bag)
    - Add **brand names** for precision
    """)

# -------------------------------------------------------------------
# 4. Search Interface
# -------------------------------------------------------------------

st.markdown('<div class="search-section">', unsafe_allow_html=True)

# Function to update the query when a button is clicked
def set_query(q):
    # Remove emoji from the query for cleaner search
    clean_q = re.sub(r'^[^\w\s]+\s*', '', q)
    st.session_state.query = clean_q

# Quick search buttons
st.markdown("### 🚀 Quick Search Options")
st.markdown("*Click any option to start searching instantly*")

# Create a more attractive button layout - 3 columns for better spacing
cols = st.columns(3)
for i, q in enumerate(PREDEFINED_QUESTIONS):
    with cols[i % 3]:
        if st.button(q, key=f"btn_{i}", help=f"Search for: {q}"):
            set_query(q)

# Search input with better styling
st.markdown("### 💬 Or Type Your Own Search")
# query = st.text_input(
#     "", 
#     placeholder="e.g., 'comfortable running shoes under ₹3,000' or 'formal black dress'",
#     value=st.session_state.query,
#     key="search_input",
#     label_visibility="collapsed"
# )

# Define a callback that runs as soon as input changes
def update_query():
    st.session_state.query = st.session_state.search_input

query = st.text_input(
    "Search",
    placeholder="e.g., 'comfortable running shoes under ₹3,000' or 'formal black dress'",
    value=st.session_state.query,
    key="search_input",
    label_visibility="collapsed",
    on_change=update_query   # 🔑 Trigger update while typing
)

# 🔮 Lightweight Auto-Suggest
if st.session_state.query:   # use session_state.query instead of query
    try:
        suggestions = requests.get(
            f"http://localhost:8000/suggest",
            params={"query": st.session_state.query}
        ).json()

        if suggestions:
            st.markdown("### 🔮 Searching Suggestions")
            for idx, s in enumerate(suggestions):
                if st.button(s, key=f"sugg_{idx}_{s}"):
                    st.session_state.query = s
                    st.rerun()
    except Exception as e:
        st.warning(f"Suggestion fetch failed: {e}")

##Auto-update session query as user types
# if st.session_state.query != query:
    # st.session_state.query = query
    # st.rerun()

# Settings row
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    if st.button("🔍 Search Now", disabled=not query, type="primary"):
        pass  # Search will trigger automatically when query changes
        
with col2:
    top_k = st.number_input(
        "Results to show", 
        min_value=1, 
        max_value=50, 
        value=10, 
        step=5,
        help="Maximum number of results to display"
    )

with col3:
    if st.button("🗑️ Clear", help="Clear search"):
        st.session_state.query = ""
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# 5. Search Functions
# -------------------------------------------------------------------

def search_api(q):
    """Makes a POST request to the search API (fetches all results)."""
    payload = {"query": q, "top_k": 200}   # fetch max possible
    r = requests.post(SEARCH_EP, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def rerank_results(hits, query):
    """Re-rank hits based on keyword overlap between query and product fields."""
    
    # Tokenize helper
    def tokenize(text):
        return re.findall(r"\w+", text.lower())
    
    query_tokens = tokenize(query)
    query_counter = Counter(query_tokens)

    for h in hits:
        combined_text = h.get("product_name", "") + " " + h.get("product_description", "")
        doc_tokens = tokenize(combined_text)
        doc_counter = Counter(doc_tokens)

        # Simple overlap score
        overlap = sum((query_counter & doc_counter).values())

        # Bonus: if query tokens appear in product_name, weight higher
        name_tokens = tokenize(h.get("product_name", ""))
        name_overlap = len(set(query_tokens) & set(name_tokens))

        h["local_score"] = overlap + 2 * name_overlap   # weights: name overlap more important

    # Sort descending by local_score
    return sorted(hits, key=lambda x: x["local_score"], reverse=True)

def highlight_text(text, query):
    """Highlights query words inside the given text using <mark>."""
    if not text:
        return ""
    query_tokens = re.findall(r"\w+", query.lower())
    highlighted = text
    for token in set(query_tokens):
        pattern = re.compile(rf"({re.escape(token)})", re.IGNORECASE)
        highlighted = pattern.sub(r"<mark>\1</mark>", highlighted)
    return highlighted

# -------------------------------------------------------------------
# 6. Search Results Display
# -------------------------------------------------------------------

if query:  # Only run a search if the query is not empty
    with st.spinner("🔍 Searching through thousands of products..."):
        try:
            res = search_api(query)
        except Exception as e:
            st.error(f"🚨 Search failed: {e}")
            st.stop()

    hits = res.get("results", [])

    # 🔑 Re-rank using query vs product_name + description
    hits = rerank_results(hits, query)

    if not hits:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px; margin: 2rem 0;">
            <h3>😔 No products found</h3>
            <p>Try adjusting your search terms or browse our quick options above</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Results counter
        st.markdown(f"""
        <div class="results-counter">
            🎯 Out of {len(hits)} products • Showing top most relevent {min(top_k, len(hits))} products
        </div>
        """, unsafe_allow_html=True)

        # Only display top_k after reranking
        hits = hits[:top_k]

        # Display results in attractive cards
        for i, h in enumerate(hits, 1):
            # Product card with better spacing
            with st.container():
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem;">
                        <div class="price-tag">₹{h['price']:,.0f}</div>
                        <div style="margin-top: 0.5rem; color: #7f8c8d; font-size: 0.9rem;">
                            Rank #{i}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Product title
                    st.markdown(f"""
                    <div class="product-title">
                        {highlight_text(h.get('product_name', 'Product Name'), query)}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Description
                    description = highlight_text(h.get('product_description', 'No description available'), query)
                    st.markdown(f'<div class="description">{description}</div>', unsafe_allow_html=True)
                    
                    # Enhanced fields as badges
                    if h.get("enhanced_fields"):
                        st.markdown("<div style='margin-top: 1rem;'>", unsafe_allow_html=True)
                        for k, v in h["enhanced_fields"].items():
                            st.markdown(f'<span class="details-badge"><strong>{k}:</strong> {v}</span>', unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
            
            # Divider between products
            st.markdown("<hr style='margin: 2rem 0; border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)

        # Advanced details in expandable section
        with st.expander("🔍 Advanced Search Details", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Search Query Analysis:**")
                st.code(query, language="text")
                
                st.markdown("**Results Summary:**")
                st.info(f"""
                - **Total matches:** {len(res.get('results', []))}
                - **Displayed:** {len(hits)}
                - **Reranked:** Yes (by relevance)
                """)
            
            with col2:
                st.markdown("**Backend Field Analysis:**")
                if res.get("field_analysis"):
                    st.json(res.get("field_analysis", {}))
                else:
                    st.write("No field analysis available")

# -------------------------------------------------------------------
# 7. Footer
# -------------------------------------------------------------------

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem; border-top: 1px solid #ecf0f1; margin-top: 3rem;">
    <p>🛍️ Smart Product Search • Powered by AI • Built with ARTISANS</p>
</div>
""", unsafe_allow_html=True)