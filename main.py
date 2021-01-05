from pathlib import Path
import datatable as dt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import time
# CUSTOM
from  dataset.SAKTDataset import SAKTDataset
from  network.SAKTModel import SAKTModel
from  tools.train import do_train

MAX_SEQ = 180
ACCEPTED_USER_CONTENT_SIZE = 4
EMBED_SIZE = 128
BATCH_SIZE = 64

def main():
    path = Path('/home/increase/deeplearning/datasets/kaggle')
    assert path.exists()
    data_types_dict = {
        'content_type_id': 'bool',
        'timestamp': 'int64',
        'user_id': 'int32',
        'content_id': 'int16',
        'answered_correctly': 'int8',
        'prior_question_elapsed_time': 'float32',
        'prior_question_had_explanation': 'bool'
    }
    target = 'answered_correctly'
    train_df = dt.fread(path / 'riiid-test-answer-prediction/train.csv',
                        columns=set(data_types_dict.keys())).to_pandas()
    print('successfully load data.')
    train_df = train_df[train_df.content_type_id == False]
    # arrange by timestamp
    train_df = train_df.sort_values(['timestamp'], ascending=True).reset_index(drop=True)
    del train_df['timestamp']
    del train_df['content_type_id']
    n_skill = train_df["content_id"].nunique()
    print("number skills:", n_skill)
    group = train_df[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(
        lambda r: (r['content_id'].values, r['answered_correctly'].values))
    del train_df
    TEST_SIZE = 0.1
    train, val = train_test_split(group, test_size=TEST_SIZE, random_state=5)
    train_dataset = SAKTDataset(train, n_skill, max_seq=MAX_SEQ)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    del train
    val_dataset = SAKTDataset(val, n_skill, max_seq=MAX_SEQ)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    del val

    # create model
    model = SAKTModel(n_skill, max_seq=MAX_SEQ, embed_dim=EMBED_SIZE, forward_expansion=1, enc_layers=2, heads=8, dropout=0.1)
    start_time = time.time()
    do_train(model,train_dataloader,val_dataloader,10, 2e-3, './ckpt/sakt.pth')
    end_time = time.time()
    print('training time cost: {} min'.format((end_time-start_time)/60.0))

if __name__=='__main__':
    main()
