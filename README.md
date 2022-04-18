# JKA-NDC
## Requirements
Please install pakeages by 
```javascript 
pip install -r requirements.txt
```
## Usage Example
Cora
```javascript 
python main.py --dataset Cora --epochs 500 --lr 5e-2 --weight_decay 5e-4 --dropout 0.1 --hidden_dim 128 --alpha 1 --tau 1.5 --k 2
```
CiteSeer
```javascript 
python main.py --dataset CiteSeer --epochs 600 --lr 3e-2 --weight_decay 1e-3 --dropout 0.5 --hidden_dim 200 --alpha 0.5 --tau 0.5 --k 3
```
PubMed
```javascript 
python main.py --dataset pubmed --runs 100 --epochs 400 --batch_size 2500 --lr 0.01 --weight_decay 0.0005 --dropout 0.2 --hidden 400 --hidden_z 400 --early_stopping 10 --alpha 10 --beta 1 --tau 0.5 --order 4
```
Amazon Computers
```javascript 
python main.py --dataset computers --runs 100 --epochs 300 --lr 0.005 --weight_decay 0.0005 --dropout 0.4 --hidden 400 --hidden_z 300 --early_stopping 10 --alpha 30 --beta 3 --tau 4 --order 6
```
Amazon Photo
```javascript 
python main.py --dataset photo --runs 100 --epochs 200 --lr 0.005 --weight_decay 0.0005 --dropout 0.5 --hidden 200 --hidden_z 200 --early_stopping 10 --alpha 25 --beta 3 --tau 4 --order 5
```

## Results
model	|Cora	|CiteSeer	|PubMed|Amazon Computers	|Amazon Photo	
------ | -----  |----------- |---|--- | -----  |
JKA-NDC|	84.1% |	74.5%|	82.1%|83.8%|	93.5% |
