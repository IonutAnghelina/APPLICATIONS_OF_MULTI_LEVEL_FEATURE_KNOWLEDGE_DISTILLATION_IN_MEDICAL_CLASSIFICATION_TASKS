from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os 

class CachedBinaryDataset(Dataset):
    def __init__(self, folder, split, transform=None):
        self.data = []
        txt=os.path.join(folder,f"{split}_labels.txt")
        with open(txt) as f:
            for line in f:
                *parts, lbl = line.strip().split()
                fp=os.path.join(folder,'images',' '.join(parts))
                try: img=Image.open(fp).convert('RGB')
                except: continue
                if transform: img=transform(img)
                self.data.append((img,int(lbl)))
                #break
    def __len__(self): return len(self.data)
    def __getitem__(self,idx): return self.data[idx]

class DomainDataset(Dataset):
    def __init__(self, base_ds, dom): self.base, self.dom = base_ds, dom
    def __len__(self): return len(self.base)
    def __getitem__(self,idx): x,y=self.base[idx]; return x,y,self.dom
