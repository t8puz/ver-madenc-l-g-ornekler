import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Örnek veri seti
data = {'Milk': [1, 0, 1, 1, 0, 1, 0],
        'Bread': [1, 1, 0, 1, 1, 0, 1],
        'Butter': [0, 1, 0, 1, 0, 1, 1],
        'Cheese': [1, 0, 1, 0, 1, 1, 0]}

df = pd.DataFrame(data)

# Sık öğe kümelerini bul
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

# Birliktelik kurallarını çıkar
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Sonuçları göster
print(frequent_itemsets)
print(rules)
