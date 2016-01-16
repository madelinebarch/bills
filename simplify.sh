cat data/bills93-113.txt | cut -f 12,27 | sed 's/\t"/\t/' | sed 's/"$//' | sed 's/??/"/g' | tr -d '?_*!>\032' | tr 'a-z' 'A-Z' | tail -n +2 > data/bills.txt 

