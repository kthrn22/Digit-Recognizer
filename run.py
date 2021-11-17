from utils import *
from modeling import *

class args:
    fold = 0
    test_split = 0.1
    hidden_size = 256
    dropout_probs = 0.4
    num_splits = 5
    batch_size = 64
    patience = 5
    epochs = 10
    batch_size = 64

df = pd.read_csv('./train.csv')
df = create_kfold(df, num_splits = args.num_splits)

df_train = df[df['kfold'] != args.fold].drop('kfold', axis = 1).reset_index(drop = True)
X_train, y_train = preprocess(df_train, mode = 'train')

df_valid = df[df['kfold'] == args.fold].drop('kfold', axis = 1).reset_index(drop = True)
X_val, y_val = preprocess(df_valid, mode = 'train')

model = Model(args.dropout_probs, args.hidden_size)
model.train((X_train, y_train), (X_val, y_val), patience = args.patience, epochs = args.epochs, 
                                batch_size = args.batch_size)
