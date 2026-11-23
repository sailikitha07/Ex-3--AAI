<H3>NAME : Cholimgapuram Sai Likitha</H3>
<H3>REGISTER NO.: 212224230046</H3>
<H3>EX. NO.3</H3>
<H3>DATE:23.11.2026</H3>
<H1 ALIGN =CENTER> Implementation of Approximate Inference in Bayesian Networks
</H1>

## Aim: 
   To construct a python program to implement approximate inference using Gibbs Sampling.</br>
## Algorithm:
   Step 1: Bayesian Network Definition and CPDs:<br>
    <ul> <li>Define the Bayesian network structure using the BayesianNetwork class from pgmpy.models.</li>
    <li>Define Conditional Probability Distributions (CPDs) for each variable using the TabularCPD class.</li>
    <li>Add the CPDs to the network.</li></ul>
    Step 2: Printing Bayesian Network Structure:<br>
    <ul><li>Print the structure of the Bayesian network using the print(network) statement.</li></ul>
   Step 3: Graph Visualization:
    <ul><li>Import the necessary libraries (networkx and matplotlib).</li>
    <li>Create a directed graph using networkx.DiGraph().</li>
    <li>Define the nodes and edges of the graph.</li>
    <li>Add nodes and edges to the graph.</li>
    <li>Optionally, define positions for the nodes.</li>
    <li>Use nx.draw() to visualize the graph using matplotlib.</li></ul>
    Step 4: Gibbs Sampling and MCMC:<br>
    <ul><li>Initialize Gibbs Sampling for MCMC using the GibbsSampling class and provide the Bayesian network.</li>
    <li>Set the number of samples to be generated using num_samples.</li></ul>
    Step 5: Perform MCMC Sampling:<br>
    <ul><li>Use the sample() method of the GibbsSampling instance to perform MCMC sampling.</li>
    <li>Store the generated samples in the samples variable.</li></ul>
    Step 6: Approximate Probability Calculation:<br>
    <ul><li>Specify the variable for which you want to calculate the approximate probabilities (query_variable).</li>
    <li>Use .value_counts(normalize=True) on the samples of the query_variable to calculate approximate probabilities.</li></ul>
    Step 7:Print Approximate Probabilities:<br>
    <ul><li>Print the calculated approximate probabilities for the specified query_variable.</li></ul>


## Program:
```
!pip install pgmpy networkx matplotlib

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import GibbsSampling
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Create Bayesian Network
model = DiscreteBayesianNetwork([
    ("Burglary", "Alarm"),
    ("Earthquake", "Alarm"),
    ("Alarm", "JohnCalls"),
    ("Alarm", "MaryCalls")
])

# Step 2: Define probabilities
b = TabularCPD("Burglary", 2, [[0.999], [0.001]])
e = TabularCPD("Earthquake", 2, [[0.998], [0.002]])

a = TabularCPD("Alarm", 2,
    [[0.999, 0.71, 0.06, 0.05],
     [0.001, 0.29, 0.94, 0.95]],
    evidence=["Burglary", "Earthquake"],
    evidence_card=[2, 2])

j = TabularCPD("JohnCalls", 2,
    [[0.95, 0.1], [0.05, 0.9]],
    evidence=["Alarm"], evidence_card=[2])

m = TabularCPD("MaryCalls", 2,
    [[0.1, 0.7], [0.9, 0.3]],
    evidence=["Alarm"], evidence_card=[2])

# Step 3: Add CPDs
model.add_cpds(b, e, a, j, m)

# Step 4: Verify model
print("Model Valid:", model.check_model())

# Step 5: Draw graph
G = nx.DiGraph()
G.add_edges_from([
    ("Burglary", "Alarm"),
    ("Earthquake", "Alarm"),
    ("Alarm", "JohnCalls"),
    ("Alarm", "MaryCalls")
])

pos = {
    "Burglary": (0, 0),
    "Earthquake": (2, 0),
    "Alarm": (1, -2),
    "JohnCalls": (0, -4),
    "MaryCalls": (2, -4)
}

nx.draw(G, pos, with_labels=True, node_color="lavender",
        node_size=2000, arrowsize=20)
plt.title("Bayesian Network")
plt.show()

# Step 6: Gibbs Sampling
sampler = GibbsSampling(model)
samples = sampler.sample(size=10000)

# Step 7: Calculate probability
result = samples["Burglary"].value_counts(normalize=True)
print("Burglary Probability:")
print(result)
```


## Output:

<img width="561" height="427" alt="image" src="https://github.com/user-attachments/assets/4a308b07-5fa7-458a-a669-6714b15b8cda" />



<img width="261" height="75" alt="image" src="https://github.com/user-attachments/assets/77eda672-55b1-4602-9413-99f2167170a6" />

## Result:
Thus, Gibb's Sampling( Approximate Inference method) is succuessfully implemented using python.
