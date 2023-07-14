# ([Time Series Domain Adaptation via Sparse Associative Structure Alignment](https://ojs.aaai.org/index.php/AAAI/article/view/16846))[AAAI2021]

## Requirements

- Python  3.7

- Pytorch 1.7

  

## Quick Start

You can run it with the following command.

```
python main.py -cuda_device 0 -dataset HHAR -batch_size 64 -seed 10 -epochs 40
```

## result
#### HAR 

| dataset | domain  | f1          |
|---------|---------|-------------|
| HAR     | 12_to_16 | 84.91813    | 
|         | 18_to_27 | 99.61904667 |  
|         | 2_to_11 | 95.89705    |     
|         | 20_to_5 | 75.57394333 |      
|         | 24_to_8 | 90.73282333 |         
|         | 28_to_27 | 95.85423333 |        
|         | 30_to_20 | 88.80905333 |        
|         | 6_to_23 | 93.36903667 |         
|         | 7_to_13 | 92.67917    |       
|         | 9_to_18 | 80.68039667 |

#### HHAR
| dataset | domain | f1          |
|---------|--------|-------------|
| HHAR    | 0_to_2 | 73.22542667 |
|         | 0_to_6 | 54.67753    |  
|         | 1_to_6 | 84.61155333 |  
|         | 3_to_8 | 77.30236333 |  
|         | 4_to_5 | 86.79520333 | 
|         | 5_to_0 | 42.24310333 |   
|         | 6_to_1 | 88.1413     |  
|         | 7_to_4 | 89.41448667 |   
|         |8_to_3	| 83.05941    |

#### WISDM
| dataset | domain   | f1          | 
|---------|----------|-------------|
| WISDM   | 17_to_23 | 51.51952333 |
|         | 2_to_11  | 82.16593333 |   
|         | 20_to_30 | 84.00848    |  
|         | 23_to_32 | 51.45548    |   
|         | 28_to_4  | 90.03818    |  
|         | 33_to_12 | 81.55814    |   
|         | 35_to_31 | 74.76046333 |   
|         | 5_to_26  | 48.03187    |   
|         | 6_to_19  | 70.86559333 |   
|         | 7_to_18  | 72.58147333 |