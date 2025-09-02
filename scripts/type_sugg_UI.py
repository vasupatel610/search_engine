# Streamlit Frontend for E-commerce Search
import os
import streamlit as st
import streamlit.components.v1 as components
import requests
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Product Search",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API configuration
# API_BASE_URL = "https://8d2a34ca7bd1.ngrok-free.app"
API_BASE_URL = os.getenv("API_BASE_URL", "http://0.0.0.0:8010")
import os

# First check Streamlit secrets, then env var, fallback to local
# API_BASE_URL = st.secrets.get("API_BASE_URL", os.getenv("API_BASE_URL", "http://127.0.0.1:8010"))

def check_api_health():
    """Check if the FastAPI backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200, response.json()
    except requests.exceptions.RequestException:
        return False, None

def get_api_stats():
    """Get statistics from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None

# Main UI
st.title("🛍️ Product Search with Smart Autocomplete")
st.markdown("*Smart suggestions for better product discovery*")

# Check API connection
api_healthy, health_data = check_api_health()
if not api_healthy:
    st.error("⚠️ **Backend API is not running!**")
    st.markdown("""
    **To start the backend server:**
    
    ```bash
    # In a separate terminal, run:
    python api_server.py
    
    # Or using uvicorn directly:
    uvicorn api_server:app --host 127.0.0.1 --port 8010 --reload
    ```
    """)
    st.stop()

# Get statistics from API
stats = get_api_stats()

# Sidebar information
st.sidebar.success("**🟢 Search Engine Connected!**")
if stats:
    st.sidebar.info(f"**Indexed Words:** {stats['indexed_words']:,}")
    st.sidebar.info(f"**Product Categories:** {stats['total_categories']}")
    st.sidebar.info(f"**Product Variations:** {stats['total_products']}")
st.sidebar.markdown("### Smart Features:")
st.sidebar.markdown("- 👥 Gender-Aware Suggestions")
st.sidebar.markdown("- 🏷️ Category-Based Filtering")
st.sidebar.markdown("- 🚫 Invalid Combination Prevention")
st.sidebar.markdown("- ✨ Context-Aware Results")

# Enhanced search interface with better styling
search_html = f"""
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
    <div id="error-message" style="display: none; padding: 10px; text-align: center; color: red;">
        ❌ Search service unavailable
    </div>
</div>
<style>
.suggestion-item {{
    padding: 12px 15px;
    cursor: pointer;
    border-bottom: 1px solid #f5f5f5;
    transition: background-color 0.2s ease;
    font-size: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.suggestion-item:hover, .suggestion-item.selected {{
    background-color: #f0f8ff;
}}
.suggestion-text {{
    color: #111;
}}
.suggestion-scope {{
    font-size: 14px;
    color: #555;
    margin-left: 8px;
    white-space: nowrap;
    background: #e8f4f8;
    padding: 2px 8px;
    border-radius: 12px;
}}
.suggestion-scope.women {{
    background: #ffe8f0;
    color: #d63384;
}}
.suggestion-scope.men {{
    background: #e8f0ff;
    color: #0066cc;
}}
.suggestion-text strong {{
    font-weight: 600;
    color: #000;
}}
</style>
<script>
const searchInput = document.getElementById("searchInput");
const suggestionsContainer = document.getElementById("suggestions-container");
const suggestionsList = document.getElementById("suggestions-list");
const suggestionDetails = document.getElementById("suggestion-details");
const loadingDiv = document.getElementById("loading");
const errorDiv = document.getElementById("error-message");
let searchTimeout;
let selectedIndex = -1;

searchInput.addEventListener("input", function() {{
    clearTimeout(searchTimeout);
    let query = this.value.trim();
    if (query.length < 1) {{
        hideSuggestions();
        return;
    }}
    loadingDiv.style.display = 'block';
    errorDiv.style.display = 'none';
    searchTimeout = setTimeout(() => fetchSuggestions(query), 250);
}});

function fetchSuggestions(query) {{
    fetch(`{API_BASE_URL}/suggestions?q=${{encodeURIComponent(query)}}&limit=8`)
        .then(response => {{
            if (!response.ok) throw new Error('API Error');
            return response.json();
        }})
        .then(data => {{
            loadingDiv.style.display = 'none';
            errorDiv.style.display = 'none';
            suggestionsList.innerHTML = "";
            
            if (data.length > 0) {{
                data.forEach((item, index) => {{
                    let div = document.createElement('div');
                    div.className = 'suggestion-item';
                    
                    let highlightedText = highlightCompletion(item.suggestion, query);
                    
                    // Add gender-specific styling
                    let scopeClass = '';
                    if (item.scope && item.scope.toLowerCase().includes('women')) {{
                        scopeClass = 'women';
                    }} else if (item.scope && item.scope.toLowerCase().includes('men')) {{
                        scopeClass = 'men';
                    }}
                    
                    div.innerHTML = `
                        <span class="suggestion-text">${{highlightedText}}</span>
                        <span class="suggestion-scope ${{scopeClass}}">${{item.scope}}</span>
                    `;
                    
                    div.dataset.suggestion = item.suggestion;
                    div.dataset.type = item.type;
                    div.dataset.scope = item.scope;
                    div.dataset.category = item.category || 'general';
                    div.dataset.gender = item.gender || 'Unisex';
                    div.addEventListener('mouseover', () => {{
                        selectedIndex = index;
                        updateSelection(suggestionsList.children);
                        updateDetails(item);
                    }});
                    div.onclick = () => {{
                        searchInput.value = item.suggestion;
                        hideSuggestions();
                    }};
                    
                    suggestionsList.appendChild(div);
                }});
                showSuggestions();
            }} else {{
                hideSuggestions();
            }}
        }})
        .catch(error => {{
            console.error('Search error:', error);
            loadingDiv.style.display = 'none';
            errorDiv.style.display = 'block';
            hideSuggestions();
        }});
}}

function highlightCompletion(text, query) {{
    const lowerText = text.toLowerCase();
    const lowerQuery = query.toLowerCase();
    const startIndex = lowerText.indexOf(lowerQuery);
    if (startIndex === 0) {{ 
        const unchangedPart = text.substring(0, query.length);
        const boldPart = text.substring(query.length);
        return `${{unchangedPart}}<strong>${{boldPart}}</strong>`;
    }}
    return text;
}}

function updateDetails(item) {{
    const iconMap = {{
        'Brand': '🏢', 'Product': '👟', 'Category': '🏷️', 'Contextual': '🎯', 'Query': '🔍'
    }};
    const icon = iconMap[item.type] || '🛍️';
    // Gender-specific emojis
    let genderIcon = '👤';
    if (item.gender === 'Women') genderIcon = '👩';
    else if (item.gender === 'Men') genderIcon = '👨';
    suggestionDetails.innerHTML = `
        <div style="text-align: center; font-size: 48px; margin-bottom: 10px;">${{icon}}</div>
        <h3 style="margin: 0; text-align: center;">${{item.suggestion}}</h3>
        <p style="color: #555; text-align: center;">Type: ${{item.type}}</p>
        <p style="color: #555; text-align: center;">${{genderIcon}} ${{item.scope || 'General'}}</p>
        ${{item.category ? `<p style="color: #888; text-align: center; font-size: 14px;">Category: ${{item.category}}</p>` : ''}}
        <div style="background: #f0f8ff; padding: 10px; border-radius: 8px; margin-top: 15px;">
            <p style="font-size:12px; color: #666; text-align: center; margin: 0;">
                ✅ Gender-appropriate suggestion<br>
                Based on product category data
            </p>
        </div>
    `;
}}

function showSuggestions() {{
    if (suggestionsList.children.length > 0) {{
        suggestionsContainer.style.display = 'block';
    }}
}}

function hideSuggestions() {{
    suggestionsContainer.style.display = 'none';
    loadingDiv.style.display = 'none';
    errorDiv.style.display = 'none';
    selectedIndex = -1;
}}

// Keyboard navigation
searchInput.addEventListener('keydown', function(e) {{
    const items = suggestionsList.children;
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
            searchInput.value = items[selectedIndex].dataset.suggestion;
            hideSuggestions();
        }}
    }} else if (e.key === 'Escape') {{
        hideSuggestions();
    }}
}});

// Corrected and completed updateSelection function
function updateSelection(items) {{
    for (let i = 0; i < items.length; i++) {{
        if (i === selectedIndex) {{
            items[i].classList.add('selected');
            const selectedItem = items[i];
            updateDetails({{
                suggestion: selectedItem.dataset.suggestion,
                type: selectedItem.dataset.type,
                scope: selectedItem.dataset.scope,
                category: selectedItem.dataset.category,
                gender: selectedItem.dataset.gender
            }});
            selectedItem.scrollIntoView({{ block: 'nearest' }});
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

components.html(search_html, height=550)