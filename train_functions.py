from MedViT.MedViT import MedViT_small
from torchvision import transforms
import os 
import torch 
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import CachedBinaryDataset
from tqdm import tqdm
from torch import optim, nn
from pytorch_grad_cam import GradCAM
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image
from models import MultiLevelJointTeacher
from datasets import DomainDataset
import torch.nn.functional as F
from models import MedViTStudent
def batch_class_weights(labels,num_classes=4):
    cnt=torch.bincount(labels,minlength=num_classes).float(); cnt[cnt==0]=1
    return (labels.size(0)/(num_classes*cnt)).to(labels.device)

def train_one_dataset(model, path, epochs=0, load=False, bs=8, LR = 1e-5, num_classes=4):

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dl_train = DataLoader(CachedBinaryDataset(path, 'train', tf), batch_size=bs, shuffle=True, num_workers=0)
    dl_test = DataLoader(CachedBinaryDataset(path, 'test', tf), batch_size=bs, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    ckpt_base = f"./checkpoints/individual_teacher_{os.path.basename(path)}"
    if load:
        ckpt_latest = ckpt_base + "_latest.pt"
        if os.path.exists(ckpt_latest):
            model.load_state_dict(torch.load(ckpt_latest))
            return model

    optimizer = optim.Adam(model.parameters(), lr=LR)
    ce_loss = nn.CrossEntropyLoss()

    target_layer = model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    for e in range(1, epochs + 1):
        model.train()
        for x, y in tqdm(dl_train, desc=f"Teacher Ep{e} [Train]"):
            x, y = x.to(device), y.to(device)
            w = batch_class_weights(y, num_classes=num_classes)
            ce_loss.weight = w
            optimizer.zero_grad()
            out = model(x)
            out = out[0] if isinstance(out, tuple) else out
            loss = ce_loss(out, y)
            loss.backward()
            optimizer.step()

        model.eval()
        def process_batch(dl):
            all_preds, all_labels = [], []
            with torch.no_grad():
                for xb, yb in dl:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    logits = logits[0] if isinstance(logits, tuple) else logits
                    preds = torch.argmax(logits, dim=1)
                    # Convert to binary: 0 vs {1,2,3}->1
                    preds_bin = (preds != 0).long()
                    labels_bin = (yb != 0).long()
                    all_preds.append(preds_bin.cpu())
                    all_labels.append(labels_bin.cpu())
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            return all_preds, all_labels

        # Compute train metrics
        train_preds, train_labels = process_batch(dl_train)
        cm_train = confusion_matrix(train_labels, train_preds, labels=[0, 1])  # 2x2
        tn, fp, fn, tp = cm_train.ravel()
        train_acc = (tn + tp) / cm_train.sum()
        train_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        train_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Compute test metrics
        test_preds, test_labels = process_batch(dl_test)
        cm_test = confusion_matrix(test_labels, test_preds, labels=[0, 1])
        tn2, fp2, fn2, tp2 = cm_test.ravel()
        test_acc = (tn2 + tp2) / cm_test.sum()
        test_prec = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0.0
        test_rec = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0.0

        # Plot 2x2 confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0', 'Pred {1,2,3}'], yticklabels=['True 0', 'True {1,2,3}'], ax=axes[0])
        axes[0].set_title(f"Epoch {e} ‐ Train Conf Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', xticklabels=['Pred 0', 'Pred {1,2,3}'], yticklabels=['True 0', 'True {1,2,3}'], ax=axes[1])
        axes[1].set_title(f"Epoch {e} ‐ Test Conf Matrix")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")
        plt.tight_layout()
        plt.show()

        # Print metrics
        print(f"Teacher Ep{e} Metrics:")
        print(f"  Train ‐ Acc: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}")
        print(f"  Test  ‐ Acc: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")

        # Grad-CAM for first train image (binary)
        first_train_img, _ = next(iter(dl_train))
        train_img = first_train_img[0].unsqueeze(0).to(device)
        train_np = first_train_img[0].permute(1,2,0).numpy()
        train_np = (train_np * 0.5) + 0.5
        train_np = np.clip(train_np, 0, 1)
        with torch.no_grad():
            logits_train = model(train_img)
            logits_train = logits_train[0] if isinstance(logits_train, tuple) else logits_train
            pred_train = logits_train.argmax(dim=1).item()
        target_cat_train = ClassifierOutputTarget(pred_train)
        cam_train = cam(input_tensor=train_img, targets=[target_cat_train])[0]
        train_overlay = show_cam_on_image(train_np, cam_train, use_rgb=True)

        # Grad-CAM for first test image
        first_test_img, _ = next(iter(dl_test))
        test_img = first_test_img[0].unsqueeze(0).to(device)
        test_np = first_test_img[0].permute(1,2,0).numpy()
        test_np = (test_np * 0.5) + 0.5
        test_np = np.clip(test_np, 0, 1)
        with torch.no_grad():
            logits_test = model(test_img)
            logits_test = logits_test[0] if isinstance(logits_test, tuple) else logits_test
            pred_test = logits_test.argmax(dim=1).item()
        target_cat_test = ClassifierOutputTarget(pred_test)
        cam_test = cam(input_tensor=test_img, targets=[target_cat_test])[0]
        test_overlay = show_cam_on_image(test_np, cam_test, use_rgb=True)

        # Display images and overlays
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 5))
        ax1.imshow(train_np); ax1.axis('off'); ax1.set_title("Train Image")
        ax2.imshow(train_overlay); ax2.axis('off'); ax2.set_title("Grad-CAM Train")
        ax3.imshow(test_np); ax3.axis('off'); ax3.set_title("Test Image")
        ax4.imshow(test_overlay); ax4.axis('off'); ax4.set_title("Grad-CAM Test")
        plt.tight_layout()
        plt.show()

        # Save checkpoint
        torch.save(model.state_dict(), f"{ckpt_base}_{e}.pt")
        torch.save(model.state_dict(), ckpt_base + "_latest.pt")

    return model

def train_joint_teacher(
    paths, teachers, fuse_indices,
    joint_ch=1024, epochs=0, load=False, bs=8,
    num_heads=8, num_attention_layers=3,
    lambda_adv=1.0, lr=1e-4, wd=1e-2, num_classes=4
):
    """
    Train a MultiLevelJointTeacher on multiple domains using a SINGLE shared
    classification head across all datasets.

    After each epoch, for each domain:
      1) Compute 4-class loss & accuracy, and show the 4x4 confusion matrix on both train and test splits.
      2) Compute binary (class 0 vs {1,2,3}) accuracy, precision, recall, and show the 2x2 confusion matrix on both splits.
      3) Overlay Grad-CAM on the first sample's image from each split.
    """


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jt = MultiLevelJointTeacher(
        teachers, fuse_indices,
        joint_ch=joint_ch, grl_lambda=lambda_adv,
        heads=num_heads, num_attention_layers=num_attention_layers,
        num_classes=num_classes, device=device
    ).to(device)

    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    ckpt = "./checkpoints/joint_teacher_latest.pt"
    if load and os.path.exists(ckpt):
        jt.load_state_dict(torch.load(ckpt, map_location=device))
        return jt

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    train_loaders, test_loaders, domain_counts = [], [], []
    for d, p in enumerate(paths):
        ds_train = DomainDataset(CachedBinaryDataset(p, 'train', transform), d)
        ds_test  = DomainDataset(CachedBinaryDataset(p, 'test', transform), d)
        train_loaders.append(DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=0))
        test_loaders.append(DataLoader(ds_test,  batch_size=bs, shuffle=False, num_workers=0))
        domain_counts.append(len(ds_train))

    joint_ds = ConcatDataset([
        DomainDataset(CachedBinaryDataset(p, 'train', transform), d)
        for d, p in enumerate(paths)
    ])
    joint_loader = DataLoader(joint_ds, batch_size=bs, shuffle=True, num_workers=0)

    total_samples = sum(domain_counts)
    counts_tensor = torch.tensor(domain_counts, dtype=torch.float32).to(device)
    domain_weights = (total_samples / counts_tensor).to(device)

    ce_domain = nn.CrossEntropyLoss(weight=domain_weights)

    optimizer = optim.AdamW(list(jt.parameters()), lr=lr, weight_decay=wd)
    num_batches = len(joint_loader)
    total_steps = max(1, epochs * num_batches)  
    step = 0

    target_layer = jt.fusions[-1]
    cam = GradCAM(model=jt, target_layers=[target_layer])

    for epoch in range(1, epochs + 1):
        jt.train()
        sum_class = sum_domain = sum_total = 0.0
        batch_count = 0

        for x, y, d in joint_loader:
            step += 1
            x, y, d = x.to(device), y.to(device), d.to(device)

            p = step / total_steps
            lambda_adv_current = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
            jt.grl.lam = lambda_adv_current

            class_logits, dom_logits, _, _ = jt(x, return_feats=True)

            weights_c = batch_class_weights(y, num_classes=num_classes).to(device)
            L_class = F.cross_entropy(class_logits, y, weight=weights_c)
            L_domain = ce_domain(dom_logits, d)

            loss = L_class + L_domain
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_class += float(L_class.item())
            sum_domain += float(L_domain.item())
            sum_total += float(loss.item())
            batch_count += 1

        avg_class = sum_class / batch_count
        avg_domain = sum_domain / batch_count
        avg_total = sum_total / batch_count

        jt.eval()
        print(f"Epoch {epoch} complete.")
        print(f"  Avg Losses: Class {avg_class:.4f}, Domain {avg_domain:.4f}, Total {avg_total:.4f}")

        # --- Per-domain evaluation (Train & Test) with the single head ---
        for split_name, loaders in [("Train", train_loaders), ("Test", test_loaders)]:
            for di, loader in enumerate(loaders):
                all_preds4, all_labels4 = [], []
                total_loss4 = 0.0
                correct4 = 0
                total4 = 0

                with torch.no_grad():
                    for xb, yb, _ in loader:
                        xb, yb = xb.to(device), yb.to(device)
                        logits4, _ = jt(xb)  # jt returns (class_logits, dom_logits) when return_feats=False
                        # use per-batch class weights for fairer averaging
                        w_c = batch_class_weights(yb, num_classes=num_classes).to(device)
                        loss4 = F.cross_entropy(logits4, yb, weight=w_c)

                        preds4 = torch.argmax(logits4, dim=1)

                        total_loss4 += loss4.item() * yb.size(0)
                        correct4 += (preds4 == yb).sum().item()
                        total4 += yb.size(0)

                        all_preds4.append(preds4.cpu())
                        all_labels4.append(yb.cpu())

                if total4 == 0:
                    print(f"  Domain {di} {split_name}: no samples.")
                    continue

                avg_loss4 = total_loss4 / total4
                acc4 = 100.0 * (correct4 / total4)
                all_preds4 = torch.cat(all_preds4).numpy()
                all_labels4 = torch.cat(all_labels4).numpy()
                cm4 = confusion_matrix(all_labels4, all_preds4, labels=list(range(num_classes)))

                # Binary metrics: class 0 vs {1,2,3}
                bin_preds = (all_preds4 != 0).astype(int)
                bin_labels = (all_labels4 != 0).astype(int)
                cm_bin = confusion_matrix(bin_labels, bin_preds, labels=[0, 1])
                tn, fp, fn, tp = cm_bin.ravel()
                acc_bin = (tn + tp) / cm_bin.sum()
                prec_bin = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec_bin = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                print(f"  Domain {di} {split_name} 4-Class Loss: {avg_loss4:.4f}, Acc: {acc4:.2f}%")
                print(f"    4-Class Confusion Matrix:\n{cm4}")
                print(f"  Domain {di} {split_name} Binary - Acc: {acc_bin:.4f}, Prec: {prec_bin:.4f}, Rec: {rec_bin:.4f}")
                print(f"    Binary Confusion Matrix:\n{cm_bin}")

                try:
                    first_x, first_y, _ = next(iter(loader))
                except StopIteration:
                    continue
                img = first_x[0].unsqueeze(0).to(device)
                img_np = first_x[0].permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 0.5) + 0.5
                img_np = np.clip(img_np, 0, 1)

                logits_gc, _ = jt(img)
                pred_gc = logits_gc.argmax(dim=1).item()
                target_gc = ClassifierOutputTarget(pred_gc)
                cam_map = cam(input_tensor=img, targets=[target_gc])[0]
                overlay = show_cam_on_image(img_np, cam_map, use_rgb=True)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
                ax1.imshow(img_np); ax1.axis('off'); ax1.set_title(f"Domain {di} {split_name} Input")
                ax2.imshow(overlay); ax2.axis('off'); ax2.set_title(f"Domain {di} {split_name} Grad-CAM")
                plt.tight_layout()
                plt.show()

        torch.save(jt.state_dict(), ckpt)

    return jt


def train_student(path: str,
                  joint_teacher: MultiLevelJointTeacher,
                  student,
                  load: bool = False,
                  fuse_indices: list = None,
                  epochs: int = 10,
                  bs: int = 8,
                  lr: float = 5e-4,
                  num_classes: int = 4,
                  temperature: float = 1.0,
                  lambda_ce: float = 1.0,
                  lambda_kld: float = 1.0,
                  lambda_fa: float = 1.0,
                  lambda_cos: float = 1.0,
                  lambda_con: float = 1.0) -> nn.Module:
    """
    Train a MedViT student with multi-level distillation:
      - Classification (CE + Focal)
      - Logit distillation (KL)
      - Feature alignment (MSE)
      - Cosine similarity
      - Contrastive KL
    """
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    train_ds = CachedBinaryDataset(path, 'train', tf)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_dl = DataLoader(CachedBinaryDataset(path, 'test', tf), batch_size=bs, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = MedViTStudent(student, num_classes=4, fuse_indices=fuse_indices,
                             joint_ch=joint_teacher.joint_ch).to(device)

    optimizer = optim.Adam(student.parameters(), lr=lr)
    ce_fn  = nn.CrossEntropyLoss()
    kld_fn = nn.KLDivLoss(reduction='batchmean')
    joint_teacher.eval()
    for param in joint_teacher.parameters():
      param.requires_grad = False
    K = len(fuse_indices)
    teacher_blocks = list(joint_teacher.fusions)
    all_counts = torch.zeros(4, dtype=torch.long)
    for data in train_ds:
        all_counts[data[1]] += 1
    total = all_counts.sum().float()
    print(f"All counts: {all_counts}\n")
    weights = total / all_counts.float()
    weights = weights.to(device)

    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    ckpt_base = f"./checkpoints/student_{os.path.basename(path)}"
    if load and os.path.exists(ckpt_base + "_latest.pt"):
        student.load_state_dict(torch.load(ckpt_base + "_latest.pt"))
        return student
    
    for ep in range(1, epochs + 1):
        student.train()
        loss_sum = 0
        ok = True
        for x, y in tqdm(train_dl, desc=f"Student Ep{ep}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Forward pass student
            s_logits, s_feats = student(x)

            # Forward pass teacher to get fusion features
            with torch.no_grad():
                _, _, teacher_feats_list, result = joint_teacher(x, return_feats=True)

            # Classification loss
            weights = batch_class_weights(y, num_classes=num_classes).to(device)
            ce_fn.weight = weights
            loss_ce = ce_fn(s_logits, y)

            # Logit distillation (KL)
            s_logp = F.log_softmax(s_logits / temperature, dim=1)
            t_soft = F.softmax(result / temperature, dim=1)
            loss_kd = kld_fn(s_logp, t_soft) * (temperature ** 2)

            # Feature-level distillation across K fusion layers
            loss_fa = 0.0
            loss_cos = 0.0
            loss_con = 0.0

            print(s_feats.size())
            print(teacher_feats_list[-K:].size())
            for sf, tf in zip(s_feats, teacher_feats_list[-K:]):
                sf = F.adaptive_avg_pool2d(sf, tf.shape[-2:])
                loss_fa += F.mse_loss(sf, tf)
                vs = sf.flatten(2).mean(-1)
                vt = tf.flatten(2).mean(-1)
                loss_cos += (1 - F.cosine_similarity(vs, vt, dim=1)).mean()
                ns = F.normalize(vs, p=2, dim=1)
                nt = F.normalize(vt, p=2, dim=1)
                ss = (ns @ ns.t()) / temperature
                st = (nt @ nt.t()) / temperature
                loss_con += kld_fn(F.log_softmax(ss, dim=1), F.softmax(st, dim=1))
            loss_fa /= K
            loss_cos /= K
            loss_con /= K

            # Total loss
            total_loss = (
                lambda_ce * loss_ce +
                lambda_kld * loss_kd +
                lambda_fa * loss_fa +
                lambda_cos * loss_cos
            )
            loss_sum += total_loss.item()
            total_loss.backward()
            optimizer.step()

        print(f"Current loss is {loss_sum}")
        student.eval()
        with torch.no_grad():
            total, correct = 0, 0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                out, _ = student(x)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            print(f"Epoch {ep}: Train Acc: {100*correct/total:.2f}%")
            total, correct = 0, 0
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                out, _ = student(x)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            print(f"Epoch {ep}: Test Acc: {100*correct/total:.2f}%")
        torch.save(student.state_dict(), f"./checkpoints/student_{os.path.basename(path)}_{ep}.pt")
        torch.save(student.state_dict(), f"./checkpoints/student_{os.path.basename(path)}_latest.pt")
    return student