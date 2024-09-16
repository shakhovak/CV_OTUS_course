import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import cv2 as cv2
import torch.nn.functional as F
import numpy as np
import os


def view_images(path_to_folder):
    images_to_display = []
    folder_names = []
    for folder in os.listdir(path_to_folder):
        for image in os.listdir(path_to_folder + "/" + folder):
            images_to_display.append(os.path.join(path_to_folder, folder,
                                                  image))
            folder_names.append(folder)
            break

    plt.figure(1, figsize=(12, 9))
    plt.axis("off")

    for i, (image, folder_name) in enumerate(zip(images_to_display,
                                                 folder_names)):
        plt.subplot(3, 4, i + 1)
        img = cv2.imread(image)
        plt.imshow(img)
        plt.title(f"{folder_name}_{img.shape}", fontsize=8)
    plt.show()


def create_embeddings(path_to_df, model, default_transform, device):
    df = pd.read_csv(path_to_df)
    model.eval()
    img_tensors = []
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            img = plt.imread(row["img_ref"])
            if len(img.shape) < 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img_transformed = default_transform(img)
            img_transformed = img_transformed.unsqueeze(0)
            img_transformed = img_transformed.to(device)
            img_tensor = model(img_transformed)
            img_tensors.append(img_tensor)
    return img_tensors


def display_k_neighbours(k, query_tensor, database_tensors, database_df):
    distances = []
    for item in database_tensors:
        euclidean_distance = F.cosine_similarity(
            query_tensor.unsqueeze(0),
            item.unsqueeze(0),
        )
        distances.append(euclidean_distance.item())
    dist_index = np.argwhere(distances)
    match = sorted(zip(distances, dist_index.tolist()), reverse=True)

    idx_sort = [x[1][0] for x in match]
    dist_sort = [x[0] for x in match]
    top_k_idx = idx_sort[0:k]
    top_k_dist = dist_sort[0:k]

    fig = plt.figure(figsize=(15, 8))
    for plot_num, (idx, score) in enumerate(zip(top_k_idx, top_k_dist)):
        ax = fig.add_subplot(2, 5, plot_num + 1)
        ax.grid("off")
        ax.axis("off")
        img = cv2.imread(database_df.iloc[idx]["img_ref"])
        ax.imshow(img[..., ::-1])
        plt.title(
            f"{database_df.iloc[idx]['target']}_score:{round(score, 4)} \n price={database_df.iloc[idx]['price']} \n {database_df.iloc[idx]['title']}",
            fontsize=8,
            loc="center",
            wrap=True,
        )
    plt.tight_layout()
    plt.show()


def display_ann_neighbours(k, query_tensor, forest, database_df):
    top_k_idx = forest.get_nns_by_vector(query_tensor.cpu().detach().tolist(), k)

    fig = plt.figure(figsize=(15, 9))

    for plot_num, idx in enumerate(top_k_idx):
        ax = fig.add_subplot(2, 5, plot_num + 1)
        ax.grid("off")
        ax.axis("off")
        img = cv2.imread(database_df.iloc[idx]["img_ref"])
        ax.imshow(img[..., ::-1])
        plt.title(
            f"{database_df.iloc[idx]['target']} \n price={database_df.iloc[idx]['price']} \n {database_df.iloc[idx]['title']}",
            fontsize=8,
            loc="center",
            wrap=True,
        )
    plt.tight_layout()
    plt.show()


def display_random_item(
    k, k_img, k_text, random_idx, method, company_data,
    company_emb, database_emb, comparable_data, forest,
    text_model, tokenizer
):
    img = cv2.imread(company_data.iloc[random_idx]["img_ref"])
    fig, ax = plt.subplots(figsize=(2, 2))
    plt.imshow(img[..., ::-1], aspect="auto")
    plt.title(
        f"{company_data.iloc[random_idx]['title']} \n price={company_data.iloc[random_idx]['price']}",
        fontsize=8,
    )
    plt.show()

    if method == 'knn':
        display_k_neighbours(
            k,
            query_tensor=company_emb[random_idx],
            database_tensors=database_emb,
            database_df=comparable_data,
        )
    elif method == 'ann':
        display_ann_neighbours(
            k,
            query_tensor=company_emb[random_idx],
            forest=forest,
            database_df=comparable_data,
        )
    else:
        display_ann_rerank(
            k_img=k_img,
            k_text=k_text,
            query_tensor=company_emb[random_idx],
            forest=forest,
            database_df=comparable_data,
            query_idx=random_idx,
            query_df=company_data,
            text_model=text_model,
            tokenizer=tokenizer,
            info_used="title_only",
        )


def calc_knn_metric(k, query_tensors, database_tensors, query_df, database_df, combined=False):
    acc_list = []
    categories_list = []
    for query_idx, tnsor in tqdm(enumerate(query_tensors), total=len(query_tensors)):
        categories_list.append(query_df.iloc[query_idx]["target"])
        distances = []
        for item in database_tensors:
            if combined:
                cosine_score = F.cosine_similarity(
                    [x[0] for x in tnsor].unsqueeze(0),
                    item.unsqueeze(0),
                )
            else:
                cosine_score = F.cosine_similarity(
                    tnsor.unsqueeze(0),
                    item.unsqueeze(0),
                )
            distances.append(cosine_score.item())
        dist_index = np.argwhere(distances)
        match = sorted(zip(distances, dist_index.tolist()), reverse=True)

        idx_sort = [x[1][0] for x in match]
        top_k_idx = idx_sort[0:k]

        pred_categories = []
        for idx in top_k_idx:
            if database_df.iloc[idx]["target"] == query_df.iloc[query_idx]["target"]:
                pred_categories.append(1)
            else:
                pred_categories.append(0)
        query_acc = sum(pred_categories) / len(pred_categories)
        acc_list.append(query_acc)
    mean_acc = sum(acc_list) / len(acc_list)
    for cat in set(categories_list):
        occurance_indices = [i for i, x in enumerate(categories_list) if x == cat]
        cat_acc = [acc_list[j] for j in occurance_indices]
        mean_cat_acc = sum(cat_acc)/len(cat_acc)
        print(f"{cat} acc at {k}: {round(mean_cat_acc, 4)}")
    print(f"Total acc at {k}: {round(mean_acc, 4)}")


def calc_ann_metric(k, forest, query_tensors, query_df, database_df, combined=False):
    acc_list = []
    categories_list = []
    for query_idx, tnsor in tqdm(enumerate(query_tensors), total=len(query_tensors)):
        categories_list.append(query_df.iloc[query_idx]["target"])
        if combined:
            indexes = forest.get_nns_by_vector([x[0] for x in tnsor.cpu().detach().tolist()], k)
        else:
            indexes = forest.get_nns_by_vector(tnsor.cpu().detach().tolist(), k)

        pred_categories = []
        for idx in indexes:
            if database_df.iloc[idx]["target"] == query_df.iloc[query_idx]["target"]:
                pred_categories.append(1)
            else:
                pred_categories.append(0)
        query_acc = sum(pred_categories) / len(pred_categories)
        acc_list.append(query_acc)
    mean_acc = sum(acc_list) / len(acc_list)
    for cat in set(categories_list):
        occurance_indices = [i for i, x in enumerate(categories_list) if x == cat]
        cat_acc = [acc_list[j] for j in occurance_indices]
        mean_cat_acc = sum(cat_acc) / len(cat_acc)
        print(f"{cat} acc at {k}: {round(mean_cat_acc, 4)}")
    print(f"Total acc at {k}: {round(mean_acc, 4)}")


def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0]


def create_text_embeddings(path_to_df, model, tokenizer,
                           info_used='title_only'):
    df = pd.read_csv(path_to_df)
    model.eval()
    text_tensors = []
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            if info_used == 'title_only':
                text = row["title"]
            elif info_used == 'title_cat':
                text = (str(row["title"]) + " " + str(row["cat_1"]) + " " + str(row["cat_2"])  + ' ' + str(row["cat_3"]))
            else:
                text = (
                    str(row["title"])
                    + " "
                    + str(row["cat_1"])
                    + " "
                    + str(row["cat_2"])
                    + " "
                    + str(row["cat_3"])
                    + " "
                    + str(row["caracteristics"])
                )
            text_emb = embed_bert_cls(text, model, tokenizer)
            text_tensors.append(text_emb)
    return text_tensors


def create_combined_embeddings(
    path_to_df, img_model, default_transform,
    text_model, tokenizer, device, info_used='title_only'
):
    df = pd.read_csv(path_to_df)
    img_model.eval()
    text_model.eval()
    combined_tensors = []
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            img = plt.imread(row["img_ref"])
            if len(img.shape) < 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img_transformed = default_transform(img)
            img_transformed = img_transformed.unsqueeze(0)
            img_transformed = img_transformed.to(device)
            img_tensor = img_model(img_transformed)
            img_tensor = img_tensor[0].unsqueeze(1)

            if info_used == 'title_only':
                text = row["title"]
            elif info_used == 'title_cat':
                text = (str(row["title"]) + " " + str(row["cat_1"]) + " " + str(row["cat_2"])  + ' ' + str(row["cat_3"]))
            else:
                text = (
                    str(row["title"])
                    + " "
                    + str(row["cat_1"])
                    + " "
                    + str(row["cat_2"])
                    + " "
                    + str(row["cat_3"])
                    + " "
                    + str(row["caracteristics"])
                )
            text_emb = embed_bert_cls(text, text_model, tokenizer)
            text_emb = text_emb.unsqueeze(1)

            combined_tensor = torch.cat([img_tensor, text_emb], dim = 0)
            combined_tensors.append(combined_tensor)

    return combined_tensors


def calc_ann_metric_rerank(k_img, k_text, forest, query_tensors, query_df, database_df, 
                           text_model, tokenizer, info_used='title_only'):
    acc_list = []
    categories_list = []
    for query_idx, tnsor in tqdm(enumerate(query_tensors), total=len(query_tensors)):
        categories_list.append(query_df.iloc[query_idx]["target"])
        indexes = forest.get_nns_by_vector(tnsor.cpu().detach().tolist(), k_img)
        if info_used == 'title_only':
            query_text = query_df.iloc[query_idx]["title"]
        elif info_used == 'title_cat':
            query_text = (
                str(query_df.iloc[query_idx]["title"])
                + " "
                + str(query_df.iloc[query_idx]["cat_1"])
                + " "
                + str(query_df.iloc[query_idx]["cat_2"])
                + " "
                + str(query_df.iloc[query_idx]["cat_3"])
            )
        else:
            query_text = (
                str(query_df.iloc[query_idx]["title"])
                + " "
                + str(query_df.iloc[query_idx]["cat_1"])
                + " "
                + str(query_df.iloc[query_idx]["cat_2"])
                + " "
                + str(query_df.iloc[query_idx]["cat_3"])
                + " "
                + str(query_df.iloc[query_idx]["caracteristics"])
                )
        query_emb = embed_bert_cls(query_text, text_model, tokenizer)
        distances = []
        for idx in indexes:
            if info_used == 'title_only':
                text = database_df.iloc[idx]["title"]
            elif info_used == 'title_cat':
                text = (
                    str(database_df.iloc[idx]["title"])
                    + " "
                    + str(database_df.iloc[idx]["cat_1"])
                    + " "
                    + str(database_df.iloc[idx]["cat_2"])
                    + " "
                    + str(database_df.iloc[idx]["cat_3"])
                )
            else:
                text = (
                    str(database_df.iloc[idx]["title"])
                    + " "
                    + str(database_df.iloc[idx]["cat_1"])
                    + " "
                    + str(database_df.iloc[idx]["cat_2"])
                    + " "
                    + str(database_df.iloc[idx]["cat_3"])
                    + " "
                    + str(database_df.iloc[idx]["caracteristics"])
                )
            text_emb = embed_bert_cls(text, text_model, tokenizer)

            cosine_score = F.cosine_similarity(
                    query_emb.unsqueeze(0),
                    text_emb.unsqueeze(0),
                )
            distances.append(cosine_score.item())
        match = sorted(zip(distances, indexes), reverse=True)
        idx_sort = [x[1] for x in match]
        top_k_idx = idx_sort[0:k_text]
        pred_categories = []
        for top_k in top_k_idx:
            if database_df.iloc[top_k]["target"] == query_df.iloc[query_idx]["target"]:
                pred_categories.append(1)
            else:
                pred_categories.append(0)
        query_acc = sum(pred_categories) / len(pred_categories)
        acc_list.append(query_acc)
    mean_acc = sum(acc_list) / len(acc_list)
    for cat in set(categories_list):
        occurance_indices = [i for i, x in enumerate(categories_list) if x == cat]
        cat_acc = [acc_list[j] for j in occurance_indices]
        mean_cat_acc = sum(cat_acc) / len(cat_acc)
        print(f"{cat} acc at {k_text}: {round(mean_cat_acc, 4)}")
    print(f"Total acc at {k_text}: {round(mean_acc, 4)}")


def display_ann_rerank(
    k_img, k_text, query_tensor, forest, database_df,
    query_idx, query_df, text_model, tokenizer, info_used="title_only"
):
    indexes = forest.get_nns_by_vector(query_tensor.cpu().detach().tolist(),
                                       k_img)
    distances = []
    if info_used == 'title_only':
        query_text = query_df.iloc[query_idx]["title"]
    elif info_used == 'title_cat':
        query_text = (
            str(query_df.iloc[query_idx]["title"])
            + " "
            + str(query_df.iloc[query_idx]["cat_1"])
            + " "
            + str(query_df.iloc[query_idx]["cat_2"])
            + " "
            + str(query_df.iloc[query_idx]["cat_3"])
        )
    else:
        query_text = (
            str(query_df.iloc[query_idx]["title"])
            + " "
            + str(query_df.iloc[query_idx]["cat_1"])
            + " "
            + str(query_df.iloc[query_idx]["cat_2"])
            + " "
            + str(query_df.iloc[query_idx]["cat_3"])
            + " "
            + str(query_df.iloc[query_idx]["caracteristics"])
        )
    query_emb = embed_bert_cls(query_text, text_model, tokenizer)
    for idx in indexes:
        if info_used == 'title_only':
            text = database_df.iloc[idx]["title"]
        elif info_used == 'title_cat':
            text = (
                str(database_df.iloc[idx]["title"])
                + " "
                + str(database_df.iloc[idx]["cat_1"])
                + " "
                + str(database_df.iloc[idx]["cat_2"])
                + " "
                + str(database_df.iloc[idx]["cat_3"])
            )
        else:
            text = (
                str(database_df.iloc[idx]["title"])
                + " "
                + str(database_df.iloc[idx]["cat_1"])
                + " "
                + str(database_df.iloc[idx]["cat_2"])
                + " "
                + str(database_df.iloc[idx]["cat_3"])
                + " "
                + str(database_df.iloc[idx]["caracteristics"])
            )
        text_emb = embed_bert_cls(text, text_model, tokenizer)
        cosine_score = F.cosine_similarity(
                query_emb.unsqueeze(0),
                text_emb.unsqueeze(0),
            )
        distances.append(cosine_score.item())
    match = sorted(zip(distances, indexes), reverse=True)
    idx_sort = [x[1] for x in match]
    top_k_idx = idx_sort[0:k_text]
    fig = plt.figure(figsize=(15, 9))

    for plot_num, ix in enumerate(top_k_idx):
        ax = fig.add_subplot(2, 5, plot_num + 1)
        ax.grid("off")
        ax.axis("off")
        img = cv2.imread(database_df.iloc[ix]["img_ref"])
        ax.imshow(img[..., ::-1])
        plt.title(
            f"{database_df.iloc[ix]['target']} \n price={database_df.iloc[ix]['price']} \n {database_df.iloc[ix]['title']}",
            fontsize=8,
            loc="center",
            wrap=True,
        )
    plt.tight_layout()
    plt.show()


def calc_ann_metric_rerank2(
    k_img,
    k_text,
    forest,
    query_tensors,
    query_df,
    database_df,
    img_model,
    default_transform,
    device
):
    acc_list = []
    categories_list = []
    for query_idx, tnsor in tqdm(enumerate(query_tensors), total=len(query_tensors)):
        categories_list.append(query_df.iloc[query_idx]["target"])
        indexes = forest.get_nns_by_vector(tnsor.cpu().detach().tolist(), k_text)
        with torch.no_grad():
            img = plt.imread(query_df.iloc[query_idx]["img_ref"])
            if len(img.shape) < 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img_transformed = default_transform(img)
            img_transformed = img_transformed.unsqueeze(0)
            img_transformed = img_transformed.to(device)
            query_emb = img_model(img_transformed)
            # query_emb_flat = [x for xs in query_emb for x in xs]
        distances = []
        for idx in indexes:
            with torch.no_grad():
                img = plt.imread(database_df.iloc[idx]["img_ref"])
                if len(img.shape) < 3:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                img_transformed = default_transform(img)
                img_transformed = img_transformed.unsqueeze(0)
                img_transformed = img_transformed.to(device)
                img_tensor = img_model(img_transformed)
                # img_tensor_flat = [x for xs in img_tensor for x in xs]

            cosine_score = F.cosine_similarity(
                query_emb.squeeze(0),
                img_tensor.squeeze(0), dim=0
            )
            distances.append(cosine_score.item())
        match = sorted(zip(distances, indexes), reverse=True)
        idx_sort = [x[1] for x in match]
        top_k_idx = idx_sort[0:k_img]
        pred_categories = []
        for top_k in top_k_idx:
            if database_df.iloc[top_k]["target"] == query_df.iloc[query_idx]["target"]:
                pred_categories.append(1)
            else:
                pred_categories.append(0)
        query_acc = sum(pred_categories) / len(pred_categories)
        acc_list.append(query_acc)
    mean_acc = sum(acc_list) / len(acc_list)
    for cat in set(categories_list):
        occurance_indices = [i for i, x in enumerate(categories_list) if x == cat]
        cat_acc = [acc_list[j] for j in occurance_indices]
        mean_cat_acc = sum(cat_acc) / len(cat_acc)
        print(f"{cat} acc at {k_img}: {round(mean_cat_acc, 4)}")
    print(f"Total acc at {k_img}: {round(mean_acc, 4)}")
