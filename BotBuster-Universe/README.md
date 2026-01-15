# BotBuster

- BotBuster: Performs bot detection for Reddit

## Interpreting BotBuster Outputs
1. BotBuster outputs are stored in the file temp/.../*_bots.json
Use the column `botornot`. The user is a bot if the `humanprob` is greater than `botprob`
2. BotSorter outputs are stored in the file temp/.../*_botsorter.json
3. BotBias outputs are stored in the file temp/.../*_bias.json

## Reference Papers:
Bot Detection and Analysis papers <a href="https://quarbby.github.io/research/botbuster_universe.html" target="_blank">here</a>.

This is the paper we will cite: 
BotBuster4Everyone: Ng, L. H. X., & Carley, K. M. (2024). Assembling a multi-platform ensemble social bot detector with applications to US 2020 elections. Social Network Analysis and Mining, 14(1), 1-16.
