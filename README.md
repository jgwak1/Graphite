## GRAPHITE: Real-Time Graph-Based Detection of Fileless Malware Attacks

This repository provides the Python implementation for **Graphite N-gram** and the dataset used in the paper:
> GRAPHITE: Real-Time Graph-Based Detection of Fileless Malware Attacks

---------------------

### Abstract

<p align="justify">
Advanced malware attacks often
employed sophisticated tactics such as DLL injection, script-based attacks, and the exploitation of zero-day vulnerabilities. As evidenced by the recent high-profile cyberattacks, these techniques have enabled attackers to infiltrate computer systems that were thought to be well-protected.
There is thus an urgent need to enhance current malware defenses with advanced Artificial Intelligence(AI) techniques that can effectively detect in real-time the elusive traces of malware attacks concealed within the extensive realm of normal
activities. This paper introduces Graphite, a graph-based approach for real-time detection of advanced malware attacks based on the event data collected from Event Tracing for Windows (ETW). Graphite first abstracts various entities and their relationships embodied within system events into computation graphs, which are amenable to graph-based machine learning methods. As a computation graph can be gigantic, making real-time malware detection inefficient, we project the graph into smaller graphlets, which are then subsequently fed into our graph-based approach to detect malicious activities. Our experimental results show that Graphite achieves 87.7% classification accuracy in offline testing and 86.7% accuracy in real-time detection.</p>



### Requirements
The codebase is implemented in Python 3.9.19. Necessary package versions for running the code are as below.
```
torch==1.13.0
torch-geometric==2.3.1
scikit-learn==1.1.1
```

### Running the code
Training and testing the **Graphite N-gram** based on the dataset provided in dataset/train and dataset/test. 
```sh
$ python3 src/main.py
```

Changing the **N** parameter for N-grams (default: 4)
```sh
$ python src/main.py --N 2
```

Changing the **pool** parameter to apply a different pooling method (default: sum)
```sh
$ python src/main.py --pool mean
```

