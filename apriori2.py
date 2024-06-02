from apyori import apriori

# Örnek veri seti
transactions = [
    ['eggs', 'bacon', 'soup'],
    ['eggs', 'bacon', 'apple'],
    ['soup', 'bacon', 'banana']
]

# Apriori algoritmasını kullanarak kuralları çıkarma
rules = apriori(transactions, min_support=0.5, min_confidence=0.7)

# Kuralları yazdırma
for rule in rules:
    print(rule)
