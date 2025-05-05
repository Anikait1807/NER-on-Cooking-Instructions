# NER-on-Cooking-Instructions

The goal of the research is to develop a domain-adapted Named Entity 
Recognition (NER) system that can accurately extract structured information 
from unstructured culinary text, enabling intelligent recipe understanding and 
downstream food-tech applications. The work can be categorised into,
 ● Models used: To build an accurate domain-specific NER system for culinary text, 
multiple transformer-based models were fine-tuned and evaluated. These include 
DeBERTa, RoBERTa, DistilRoBERTa, and spaCy’s transformer pipeline. All 
models were trained using the BIO tagging scheme, which enables precise span 
boundary recognition—essential for handling multi-word entities such as ingredient 
names. 
● Data Augmentation and Analysis: To improve model robustness and 
generalization, data augmentation techniques were employed, such as entity 
replacement and random oversampling, tailored to ingredient-centric contexts. A 
thorough data analysis was also conducted to examine entity distribution and label 
frequency, helping identify imbalances and improve annotation consistency. These 
strategies led to noticeable improvements in performance across all models, 
particularly in handling noisy and varied recipe text.
 The main objective was to design a robust NER model tailored to the culinary domain, 
capable of identifying key entities such as ingredients, quantities, units, sizes, and 
dry/fresh state within informal cooking instructions.
