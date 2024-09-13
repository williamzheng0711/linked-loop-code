# Linked-loop Code

This is the codebase to the papers: **"Coding for the unsourced A-channel with erasures: the linked loop code"** (https://ieeexplore.ieee.org/abstract/document/10289955, EUSIPCO 2023) and **"Coding for the Unsourced B-Channel with Erasures: Enhancing the Linked Loop Code"** (https://ieeexplore.ieee.org/abstract/document/10445848, ICASSP 2024). 

## Where to run the codes?
To start a simulation, please use go to the **Develop** branch, and the folder **playground**, then use the command **python simple_launcher.py --Ka=100 --pe=0.05 --L=16 --sic=1 --M=2 --ctype=B** to launch a simulation. The name of the parameters should be self-explanatory.
* **K** (>1) denotes the number of active users.
* **pe** (positive number between [0,1]) denotes the erasure probability, when $$p_e=0$$ the channel is clean, and when $$p_e=1$$ the channel is erasing everything.
* **L** (15 or 16)denotes the number of sections.
* **sic** (0 or 1) indicates whether to apply successive interference cancallation.
* **M** (2 or 3) denotes the window size -- how many previous sections a parity portion of any section should base on.
* **ctype** ("A" or "B") determines the channel type, deciding whether to simulate on A-channel or B-channel. 

## How to read the results? 
Some result like the following is expected to appear, where we applied a 3-phased recovery step. 
In specific, this demostration result says we managed to recover 92 true positives out of 100 transmitted messages. And 1 false alarm (in this simulation). 
* Phase 1 try to find ALL parity consistent paths that suffers NO erasures. 
* Phase 2 try to find ALL (the remaining) parity consistent paths that suffers 1 erasures.
* Phase 3 try to find SOME (the remaining) parity consistent paths that suffers 2 (E.g., not happening in consecutive sections) erasures.

```md
####### Start Rocking ######## K=100 and p_e= 0.05 and L= 16 and M= 2 and ctype: B
 -- Decoding phase 1 now starts.
 99%|███████████████████████████████████████████████████████████████| 95/96 [00:02<00:00, 37.04it/s]
 | Time of phase 1 (LLC): 2.573298931121826
 | In phase 1 linked loop Code decodes 42 true message out of 42
 -Phase 1 Done.

 -- Decoding phase 2 now starts.
 98%|██████████████████████████████████████████████████████████████| 53/54 [00:29<00:00,  1.81it/s]
 | Time of phase 2.1 29.349550008773804
 | In phase 2.1 Linked-loop Code decodes 35 true message out of 35
 95%|██████████████████████████████████████████████████████████████| 19/20 [00:00<00:00, 42.04it/s]
 | Time of phase 2.2 0.4530029296875
 | In phase 2.2 Linked-loop Code decodes 1 true message out of 1
 | In phase up-to-phase 2 Linked-loop Code decodes 78 true message out of 78
 -Phase 2 is done. 

 -- Decoding phase 3 now starts.
 95%|██████████████████████████████████████████████████████████████  | 18/19 [00:15<00:00,  1.14it/s]
 | Time of phase 3.1 15.845634698867798
 | In phase 3.1 Linked-loop Code decodes 12 true message out of 13
 88%|█████████████████████████████████████████████████████████████| 7/8 [00:02<00:00,  2.35it/s]
 | Time of phase 3.2 2.9842469692230225
 | In phase 3.2 Linked-loop Code decodes 2 true message out of 2
 80%|█████████████████████████████████████████████████████████████| 4/5 [00:00<00:00, 109.37it/s]
 | Time of phase 3.3 0.03735232353210449
 | In phase up-to-phase 3 Linked-loop Code decodes 92 true message out of 93
 -Phase 3 is done, this simulation terminates.
```
