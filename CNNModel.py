from tqdm import tqdm

from models import *

tqdm.pandas()
from sklearn.model_selection import StratifiedKFold
from transformers import *

import torch.utils.data
import torch.nn.functional as F

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from utils import *

from vncorenlp import VnCoreNLP

from transformers.modeling_utils import *


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """

    string = string.replace("\\", "")
    # string = bytes(string, 'ascii','ignore')
    string = bytes(string, 'utf-8', 'ignore')
    string = re.sub(b'\'', b'', string)
    string = re.sub(b'\"', b'', string)
    return string.strip().lower()


class SemtimentNetwork(nn.Module):
    def __init__(self, number_of_class):
        super(SemtimentNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1000, kernel_size=5, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1000, out_channels=256, kernel_size=5, padding=1)

        self.conv3 = nn.Conv1d(in_channels=1, out_channels=1000, kernel_size=4, padding=1)
        self.conv4 = nn.Conv1d(in_channels=1000, out_channels=256, kernel_size=4, padding=1)

        self.conv5 = nn.Conv1d(in_channels=1, out_channels=1000, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(in_channels=1000, out_channels=256, kernel_size=3, padding=1)

        self.conv7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, padding=1)

        self.class_number = number_of_class

        self.fc1 = nn.Linear(in_features=768, out_features=256)

    def forward(self, x, batch=1):
        # input layer

        # first hidden layer
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = F.relu(x1)
        x1 = F.max_pool1d(x1, kernel_size=5)

        x2 = self.conv3(x)
        x2 = self.conv4(x2)
        x2 = F.relu(x2)
        x2 = F.max_pool1d(x2, kernel_size=5)

        x3 = self.conv3(x)
        x3 = self.conv6(x3)
        x3 = F.max_pool1d(x3, kernel_size=5)

        # concatenate layer
        x = torch.cat((x2, x1, x3), dim=0)

        # second hidden layer
        x = self.conv7(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=5)
        # third hidden layer
        x = self.conv7(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=30)

        # x = self.flatten(x)
        x = x.reshape((batch, 768))
        x = self.fc1(x)

        return x

        return nn.functional.softmax(x, dim=(batch, self.class_number))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train_path', type=str, default='./data/train.csv')
    parser.add_argument('--dict_path', type=str, default="./phobert/dict.txt")
    parser.add_argument('--rdrsegmenter_path', type=str, required=True)
    parser.add_argument('--max_sequence_length', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--accumulation_steps', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=69)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--ckpt_path', type=str, default='./models')
    parser.add_argument('--bpe-codes', default="./phobert/bpe.codes", type=str, help='path to fastBPE BPE')

    seed_everything(69)

    args = parser.parse_args()
    bpe = fastBPE(args)

    rdrsegmenter = VnCoreNLP(args.rdrsegmenter_path, annotators="wseg", max_heap_size='-Xmx500m')

    sen = SemtimentNetwork(2)
    if(os.path.isfile("./models/Cnn.bin")):
        sen.load_state_dict(torch.load("./models/Cnn.bin"))


    sen.cuda()
    sen.type()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vocab = Dictionary()
    vocab.add_from_file(args.dict_path)

    # Load training data
    train_df = pd.read_csv(args.train_path, sep='\t').fillna("###")
    """
    Chuẩn hóa lại chuỗi dùng tokenize sản phẩm -> sản_phẩm 
    """
    train_df.text = train_df.text.progress_apply(
        lambda x: ' '.join([' '.join(sent) for sent in rdrsegmenter.tokenize(x)]))
    y = train_df.label.values
    """
    Trả về ma trận vector embedding [dòng, 256]
    """
    X_train = convert_lines(train_df, vocab, bpe, args.max_sequence_length)
    # cls
    # sen.cuda()
    print(device)

    criterion = nn.CrossEntropyLoss()

    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(X_train, y))

    for fold, (train_idx, val_idx) in enumerate(splits):
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[train_idx], dtype=torch.long),
                                                       torch.tensor(y[train_idx], dtype=torch.long))
        valid_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[val_idx], dtype=torch.long),
                                                       torch.tensor(y[val_idx], dtype=torch.long))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

        num_train_optimization_steps = int(args.epochs * len(train_df) / args.batch_size / args.accumulation_steps)

        optimizer = AdamW(sen.parameters(), lr=args.lr,
                          correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False

        for epoch in range(args.epochs):  # loop over the dataset multiple times
            running_loss = 0.0

            val_preds = None

            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].cuda(), data[1].cuda()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                # pdb.set_trace()

                outputs = sen(torch.unsqueeze(inputs, 1).float(), len(inputs))

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                print('[%d] loss: %.3f' % (epoch, running_loss / (i + 1)))
            running_loss = 0.0

        torch.save(sen.state_dict(), "./models/Cnn.bin")
