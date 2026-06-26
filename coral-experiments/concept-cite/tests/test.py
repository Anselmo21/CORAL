import os
os.environ["HF_TOKEN"] = "<HF_TOKEN>"

from context_cite import ContextCiter, ConceptCriteriaPartitioner, CONCEPT_CRITERIA_PROMPT_TEMPLATE

question = "What is the best university?"
concepts = ["research", 
            "teaching quality", 
            "employability", 
            "student life", 
            "campus facilities", 
            "alumni network", 
            "location", 
            "diversity and inclusion"]
# """
# follows the order of concepts in the prompt...
#       Score            Source
# 0  0.672970     employability
# 1  0.526320  teaching quality
# 2  0.038504          research

#       Score            Source
# 0  0.683547  teaching quality
# 1  0.609759     employability
# 2 -0.036690          research

#       Score            Source
# 0  0.628894     employability
# 1  0.544155  teaching quality
# 2  0.082610          research
# """
answers = ["University of Oxford", "MIT", "Harvard University"]
sources_block = """
source 1: QS World University Rankings 2024: 1. The University of Oxford, 2. The University of Cambridge, 3. Harvard University, 4. MIT, 5. The University of Stanford, ...
source 2: THE World University Rankings 2024: 1. The University of Oxford, 2. Stanford University, 3. MIT, 4. Harvard University, 5. University of Cambridge, ...
source 3: ARWU (2024): 1. Harvard University, 2. Stanford University, 3. Massachusetts Institute of Technology, 4. University of Cambridge, 5. University of California, Berkeley
source 4: U.S. News Best Global Universities (2024-2025): 1. Harvard University, 2. Massachusetts Institute of Technology, 3. Stanford University, 4. University of Oxford, 5. University of California, Berkeley
source 5: QS World University Rankings 2025: 1. Massachusetts Institute of Technology, 2. Imperial College London, 3. University of Oxford, 4. Harvard University, 5. University of Cambridge
"""

partitioner = ConceptCriteriaPartitioner(question, concepts, sources_block, answers=answers)
canonical = partitioner.context  # same as ConceptCriteriaPartitioner.canonical_context(...)

cc = ContextCiter.from_pretrained(
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "meta-llama/Llama-3.1-8B-Instruct",
    context=canonical,
    query="",  # unused with CONCEPT_CRITERIA_PROMPT_TEMPLATE
    partitioner=partitioner,
    prompt_template=CONCEPT_CRITERIA_PROMPT_TEMPLATE,
    criteria_baseline="empty",  # or "full", or None
    num_ablations=32,
)


print(cc.response)
# print(cc.prompt_template)

attributions = cc.get_attributions(as_dataframe=True, top_k=8)
df = attributions.data
print(df)

# let's revert this back to a mapping of [0, 1]
# they apply a mapping of log(p / (1-p)) to the scores, so we can apply the inverse logit function to get back to [0, 1]
import numpy as np
def inverse_logit(x):
    return 1 / (1 + np.exp(-x))

df["InverseScore"] = df["Score"].apply(inverse_logit)
# print(df)/

# let's try min-max normalization to get the scores back to [0, 1]
df["MinMaxScore"] = (df["Score"] - df["Score"].min()) / (df["Score"].max() - df["Score"].min())
print(df)

# solver =cc.last_fitted_solver
# print(solver.model_)