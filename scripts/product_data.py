import pandas as pd
import uuid
import random

# Define categories with realistic brand mappings
categories = {
    "Clothing": {
        "types": ["Shirt", "T-shirt", "Jeans", "Jacket", "Dress", "Kurta", "Saree", "Skirt", "Sweater", "Hoodie"],
        "brands": ["Zara", "H&M", "Levis", "Nike", "Adidas"]
    },
    "Footwear": {
        "types": ["Sneakers", "Sandals", "Heels", "Boots", "Loafers", "Slippers", "Running Shoes"],
        "brands": ["Nike", "Adidas", "Puma", "Reebok", "Bata"]
    },
    "Accessories": {
        "types": ["Watch", "Belt", "Handbag", "Wallet", "Scarf", "Sunglasses", "Hat", "Cap", "Jewelry", "Backpack"],
        "brands": ["Ray-Ban", "Gucci", "Fossil", "Tommy Hilfiger", "Michael Kors"]
    }
}

# Restrict materials per category
category_materials = {
    "Clothing": ["Cotton", "Polyester", "Denim", "Silk", "Wool"],
    "Footwear": ["Leather", "Synthetic", "Polyester", "Canvas", "Suede"],
    "Accessories": ["Leather", "Synthetic", "Cotton", "Wool", "Metal"]
}

# Define relevant occasions for each category
category_occasions = {
    "Clothing": {
        "Shirt": ["Formal", "Casual", "Office"],
        "T-shirt": ["Casual", "Sports", "Travel"],
        "Jeans": ["Casual", "Travel"],
        "Jacket": ["Casual", "Winter", "Travel"],
        "Dress": ["Party", "Formal", "Wedding"],
        "Kurta": ["Festive", "Traditional", "Casual"],
        "Saree": ["Wedding", "Festive", "Formal"],
        "Skirt": ["Casual", "Party"],
        "Sweater": ["Winter", "Casual"],
        "Hoodie": ["Casual", "Travel", "Sports"]
    },
    "Footwear": {
        "Sneakers": ["Casual", "Sports", "Travel"],
        "Sandals": ["Casual", "Beach", "Summer"],
        "Heels": ["Party", "Formal", "Wedding"],
        "Boots": ["Winter", "Travel", "Casual"],
        "Loafers": ["Office", "Formal", "Casual"],
        "Slippers": ["Casual", "Home", "Beach"],
        "Running Shoes": ["Sports", "Gym", "Running"]
    },
    "Accessories": {
        "Watch": ["Office", "Formal", "Casual"],
        "Belt": ["Formal", "Office", "Casual"],
        "Handbag": ["Casual", "Party", "Formal"],
        "Wallet": ["Casual", "Office"],
        "Scarf": ["Winter", "Casual", "Formal"],
        "Sunglasses": ["Travel", "Beach", "Casual"],
        "Hat": ["Summer", "Travel", "Beach"],
        "Cap": ["Casual", "Sports"],
        "Jewelry": ["Wedding", "Party", "Festive"],
        "Backpack": ["Travel", "Casual", "College"]
    }
}

# Define relevant age groups
category_age_groups = {
    "Clothing": ["Teens", "Adults", "All Ages"],
    "Footwear": ["Kids", "Teens", "Adults", "All Ages"],
    "Accessories": ["Teens", "Adults", "All Ages"]
}

colors = ["Red", "Blue", "Black", "White", "Green", "Yellow", "Brown", "Grey", "Pink", "Purple"]
sizes_clothing = ["XS", "S", "M", "L", "XL", "XXL"]
sizes_footwear = [f"{i}UK" for i in range(5, 12)]
sizes_accessories = ["One Size"]

# Varied description templates
description_templates = [
    "Elegant {color} {material} {product_type} by {brand}, designed for premium comfort.",
    "Trendy {color} {product_type} made with high-quality {material}, a must-have from {brand}.",
    "Experience the perfect blend of style and durability with this {color} {product_type} crafted in {material} by {brand}.",
    "Classic {brand} {product_type} in {color} {material}, ideal for daily wear.",
    "Exclusive {color} {product_type} from {brand}, featuring fine {material} craftsmanship.",
    "Comfortable and chic {material} {product_type} by {brand} in {color}, perfect for casual outings.",
    "Premium {color} {product_type} with {material} finish from {brand}, elevating your wardrobe.",
    "Sophisticated {brand} {product_type} in {color}, tailored from {material} for a modern look.",
    "Versatile {color} {material} {product_type} by {brand}, suitable for both work and leisure.",
    "High-fashion {product_type} from {brand} in {color}, made of {material} for a standout style."
]

# Generate unique products
products = []
used_combinations = set()

while len(products) < 100:
    category = random.choice(list(categories.keys()))
    product_type = random.choice(categories[category]["types"])
    brand = random.choice(categories[category]["brands"])
    color = random.choice(colors)
    material = random.choice(category_materials[category])

    # Ensure uniqueness
    combo = (category, product_type, brand, color, material)
    if combo in used_combinations:
        continue
    used_combinations.add(combo)

    # Sizes
    if category == "Clothing":
        size = random.choice(sizes_clothing)
    elif category == "Footwear":
        size = random.choice(sizes_footwear)
    else:
        size = "One Size"

    # Prices
    price_listed = round(random.uniform(499, 9999), 2)
    discount = random.uniform(0.1, 0.4)
    price_final = round(price_listed * (1 - discount), 2)

    # Stock
    stock_status = random.choice([0, 1])

    # Occasion (relevant to type)
    occasion = random.choice(category_occasions[category][product_type])

    # Age group (category-level relevance)
    age_group = random.choice(category_age_groups[category])

    # Product details
    product_id = str(uuid.uuid4())
    product_name = f"{color} {brand} {product_type}"
    product_image_url = f"https://picsum.photos/seed/{product_id[:8]}/200"

    # Varied description
    template = random.choice(description_templates)
    description = template.format(
        color=color.lower(),
        material=material.lower(),
        product_type=product_type.lower(),
        brand=brand
    )

    products.append({
        "product_id": product_id,
        "product_name": product_name,
        "product_category": category,
        "brand": brand,
        "size": size,
        "color": color,
        "material": material,
        "occasion": occasion,
        "age_group": age_group,
        "price_listed": price_listed,
        "price_final": price_final,
        "stock_status": stock_status,
        "product_image_url": product_image_url,
        "product_description": description
    })

# Save to CSV
df = pd.DataFrame(products)
df.to_csv("fashion_products_100_with_occasion_age.csv", index=False)

print("✅ Dataset generated: fashion_products_100_with_occasion_age.csv")



# import pandas as pd
# import uuid
# import random

# # Define categories with realistic brand mappings
# categories = {
#     "Clothing": {
#         "types": ["Shirt", "T-shirt", "Jeans", "Jacket", "Dress", "Kurta", "Saree", "Skirt", "Sweater", "Hoodie"],
#         "brands": ["Zara", "H&M", "Levis", "Nike", "Adidas"]
#     },
#     "Footwear": {
#         "types": ["Sneakers", "Sandals", "Heels", "Boots", "Loafers", "Slippers", "Running Shoes"],
#         "brands": ["Nike", "Adidas", "Puma", "Reebok", "Bata"]
#     },
#     "Accessories": {
#         "types": ["Watch", "Belt", "Handbag", "Wallet", "Scarf", "Sunglasses", "Hat", "Cap", "Jewelry", "Backpack"],
#         "brands": ["Ray-Ban", "Gucci", "Fossil", "Tommy Hilfiger", "Michael Kors"]
#     }
# }

# # Restrict materials per category to avoid mismatches
# category_materials = {
#     "Clothing": ["Cotton", "Polyester", "Denim", "Silk", "Wool"],
#     "Footwear": ["Leather", "Synthetic", "Polyester", "Canvas", "Suede"],
#     "Accessories": ["Leather", "Synthetic", "Cotton", "Wool", "Metal"]
# }

# colors = ["Red", "Blue", "Black", "White", "Green", "Yellow", "Brown", "Grey", "Pink", "Purple"]
# sizes_clothing = ["XS", "S", "M", "L", "XL", "XXL"]
# sizes_footwear = [f"{i}UK" for i in range(5, 12)]
# sizes_accessories = ["One Size"]

# # Varied description templates
# description_templates = [
#     "Elegant {color} {material} {product_type} by {brand}, designed for premium comfort.",
#     "Trendy {color} {product_type} made with high-quality {material}, a must-have from {brand}.",
#     "Experience the perfect blend of style and durability with this {color} {product_type} crafted in {material} by {brand}.",
#     "Classic {brand} {product_type} in {color} {material}, ideal for daily wear.",
#     "Exclusive {color} {product_type} from {brand}, featuring fine {material} craftsmanship.",
#     "Comfortable and chic {material} {product_type} by {brand} in {color}, perfect for casual outings.",
#     "Premium {color} {product_type} with {material} finish from {brand}, elevating your wardrobe.",
#     "Sophisticated {brand} {product_type} in {color}, tailored from {material} for a modern look.",
#     "Versatile {color} {material} {product_type} by {brand}, suitable for both work and leisure.",
#     "High-fashion {product_type} from {brand} in {color}, made of {material} for a standout style."
# ]

# # Generate unique products
# products = []
# used_combinations = set()

# while len(products) < 100:
#     category = random.choice(list(categories.keys()))
#     product_type = random.choice(categories[category]["types"])
#     brand = random.choice(categories[category]["brands"])
#     color = random.choice(colors)
#     material = random.choice(category_materials[category])

#     # Ensure uniqueness
#     combo = (category, product_type, brand, color, material)
#     if combo in used_combinations:
#         continue
#     used_combinations.add(combo)

#     # Sizes
#     if category == "Clothing":
#         size = random.choice(sizes_clothing)
#     elif category == "Footwear":
#         size = random.choice(sizes_footwear)
#     else:
#         size = "One Size"

#     # Prices
#     price_listed = round(random.uniform(499, 9999), 2)
#     discount = random.uniform(0.1, 0.4)
#     price_final = round(price_listed * (1 - discount), 2)

#     # Stock
#     stock_status = random.choice([0, 1])

#     # Product details
#     product_id = str(uuid.uuid4())
#     product_name = f"{color} {brand} {product_type}"
#     product_image_url = f"https://picsum.photos/seed/{product_id[:8]}/200"

#     # Varied description
#     template = random.choice(description_templates)
#     description = template.format(
#         color=color.lower(),
#         material=material.lower(),
#         product_type=product_type.lower(),
#         brand=brand
#     )

#     products.append({
#         "product_id": product_id,
#         "product_name": product_name,
#         "product_category": category,
#         "brand": brand,
#         "size": size,
#         "color": color,
#         "material": material,
#         "price_listed": price_listed,
#         "price_final": price_final,
#         "stock_status": stock_status,
#         "product_image_url": product_image_url,
#         "product_description": description
#     })

# # Save to CSV
# df = pd.DataFrame(products)
# df.to_csv("fashion_products_100_clean.csv", index=False)

# print("✅ Dataset generated: fashion_products_100_clean.csv")
