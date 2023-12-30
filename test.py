import os
import h5py
import PIL.Image
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import re
from transformers import BlipProcessor, BlipForConditionalGeneration

# images = os.listdir('./flickr 30k/flickr30k-images')
# data = {'ID': [], 'caption': []}
# for i in tqdm(range(10000)):
#     data['ID'].append(images[i])
#     # data['Image'].append(PIL.Image.open(os.path.join('./flickr 30k/flickr30k-images', images[i])))
#     caption = ''
#     with open('./flickr 30k/flickr30k/results_20130124.token', 'r') as file:
#         for content in file:
#             pairs = re.split(r'(\d+\.jpg#[0-9]+)', content)
#             # print(pairs)
#             ids = pairs[1].split('#')
#             # print(ids)
#             if ids[0] == images[i]:
#                 pairs[2] = pairs[2].replace('\n', '')
#                 pairs[2] = pairs[2].replace('\t', '')
#                 pairs[2] = pairs[2].replace('  ', '')
#                 caption += pairs[2]
#                 break
#     data['caption'].append(caption)
# print(data)

image_data_df = pd.read_csv('data_.csv')
image_data_df = image_data_df[0:2500]
# image_data_df.to_csv('data_.csv', index=False)
# print(image_data_df.head())
index = 'clip1.h5'
h5f = h5py.File(index, 'r')
img_feats = h5f['dataset1'][:]
text_feats = h5f['dataset2'][:]
ids = h5f['dataset3'][:]
for i in range(len(ids)):
    ids[i] = ids[i].decode('utf-8')


def get_image(ids):
    return PIL.Image.open(os.path.join('./flickr 30k/flickr30k-images', ids))


def get_img_feature(id_s):
    global ids
    for i in range(len(ids)):
        if id_s == ids[i]:
            return img_feats[i]
    return None


def get_text_feature(id_s):
    global ids
    for i in range(len(ids)):
        if id_s == ids[i]:
            return text_feats[i]
    return None


image_data_df['image'] = image_data_df['ID'].apply(get_image)
image_data_df['img_embeddings'] = image_data_df['ID'].apply(get_img_feature)
image_data_df['text_embeddings'] = image_data_df['ID'].apply(get_text_feature)
print(image_data_df.shape)
print(image_data_df.head())


def get_model_info(model_ID, device):
    # 将模型保存到设备上
    model = CLIPModel.from_pretrained(model_ID).to(device)
    # 获取处理器
    processor = CLIPProcessor.from_pretrained(model_ID)
    # 获取分词器
    tokenizer = CLIPTokenizer.from_pretrained(model_ID)
    # 返回模型、处理器和分词器
    return model, processor, tokenizer


# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
model_ID = "clip-model"
model, processor, tokenizer = get_model_info(model_ID, device)


def get_single_text_embedding(text):
    # print(text)
    # print(len(text))
    inputs = tokenizer(text, return_tensors="pt").to(device)
    text_embeddings = model.get_text_features(**inputs)
    # 将嵌入转换为NumPy数组
    embedding_as_np = text_embeddings.cpu().detach().numpy()
    return embedding_as_np


def get_all_text_embeddings(df, text_col):
    df["text_embeddings"] = df[str(text_col)].apply(get_single_text_embedding)
    return df


def get_single_image_embedding(my_image):
    # print(my_image)
    image = processor(
        text=None,
        images=my_image,
        return_tensors="pt"
    )["pixel_values"].to(device)
    embedding = model.get_image_features(image)
    # 将嵌入转换为NumPy数组
    embedding_as_np = embedding.cpu().detach().numpy()
    # print(embedding_as_np.shape)
    return embedding_as_np


def get_all_images_embedding(df, img_column):
    df["img_embeddings"] = df[str(img_column)].apply(get_single_image_embedding)
    return df


# image_data_df = get_all_images_embedding(image_data_df, "image")
# image_data_df = get_all_text_embeddings(image_data_df, "caption")
# print(image_data_df.head())
# image_feature = image_data_df['img_embeddings'].tolist()
# text_feature = image_data_df['text_embeddings'].tolist()
# images = image_data_df['ID'].tolist()
# h5f = h5py.File('clip1.h5', 'w')
# h5f.create_dataset('dataset1', data=image_feature)
# h5f.create_dataset('dataset2', data=text_feature)
# h5f.create_dataset('dataset3', data=images)


def plot_images(images):
    for image in images:
        plt.imshow(image)
        plt.show()


def plot_images_by_side(top_images):
    index_values = list(top_images.index.values)
    list_images = [top_images.iloc[idx].image for idx in index_values]
    list_captions = [top_images.iloc[idx].caption for idx in index_values]
    similarity_score = [top_images.iloc[idx].cos_sim for idx in index_values]

    n_row = n_col = 2

    _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax, caption, sim_score in zip(list_images, axs, list_captions, similarity_score):
        ax.imshow(img)
        sim_score = 100 * float("{:.2f}".format(sim_score))
        ax.title.set_text(f"Caption: {caption}\nSimilarity: {sim_score}%")
    plt.show()


# 仅图像或仅文字检索top_K
def get_top_N_images(query, data, top_K=4, search_criterion="text"):
    """
    检索与查询相似的前K篇文章（默认值为4）
    """
    # 文本到图像搜索
    if (search_criterion.lower() == "text"):
        query_vect = get_single_text_embedding(query)
    # 图像到图像搜索
    else:
        query_vect = get_single_image_embedding(query)
    # 相关列
    revevant_cols = ["caption", "image", "cos_sim"]
    # 运行相似度搜索
    data["cos_sim"] = data["img_embeddings"].apply(lambda x: cosine_similarity(query_vect, x))
    data["cos_sim"] = data["cos_sim"].apply(lambda x: x[0][0])
    """
    按余弦相似度列降序排序
    在排序时从1开始，以排除与自身的相似度（因为它始终为1）
    """
    most_similar_articles = data.sort_values(by='cos_sim', ascending=False)[1:top_K + 1]
    print(most_similar_articles)
    return most_similar_articles[revevant_cols].reset_index()


def plot_single_image(image, caption):
    plt.figure(1)
    plt.title(caption)
    plt.imshow(image)
    # plt.show()


def blip(img_path):
    processor = BlipProcessor.from_pretrained("./blip")
    model = BlipForConditionalGeneration.from_pretrained("./blip", torch_dtype=torch.float16).to("cuda")

    raw_image = Image.open(img_path).convert('RGB')

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    return processor.decode(out[0], skip_special_tokens=True)


# 混合检索top_K(将图像特征和文字特征结合)
def get_top_N_images_mix(query_image, caption, data, top_K=4):
    """
    检索与查询相似的前K篇文章（默认值为4）
    """
    # 文本到图像搜索
    text_embedding = get_single_text_embedding(caption)
    # 图像到图像搜索
    image_embedding = get_single_image_embedding(query_image)
    # 相关列
    revevant_cols = ["caption", "image", "cos_sim"]
    # 运行相似度搜索
    data["cos_sim_img"] = data["img_embeddings"].apply(lambda x: cosine_similarity(image_embedding, x))
    data["cos_sim_img"] = data["cos_sim_img"].apply(lambda x: x[0][0])
    data["cos_sim_text"] = data["text_embeddings"].apply(lambda x: cosine_similarity(text_embedding, x))
    data["cos_sim_text"] = data["cos_sim_text"].apply(lambda x: x[0][0])
    data['cos_sim'] = 0.7 * data['cos_sim_img'] + 0.3 * data['cos_sim_text']
    """
    按余弦相似度列降序排序
    在排序时从1开始，以排除与自身的相似度（因为它始终为1）
    """
    most_similar_articles = data.sort_values(by='cos_sim', ascending=False)[0:top_K]
    print(most_similar_articles)
    return most_similar_articles[revevant_cols].reset_index()


# query_caption = image_data_df.iloc[8].caption
# top_images = get_top_N_images(query_caption, image_data_df)
# print("Query: {}".format(query_caption))
# plot_images_by_side(top_images)
# query_image = image_data_df.iloc[28].image
# top_images = get_top_N_images(query_image, image_data_df, search_criterion="image")


# 输入图片路径
ime_path = ''
query_image = PIL.Image.open(ime_path)
# 利用blip模型生成对该图片的文字描述
caption = blip(ime_path)
plot_single_image(query_image, caption)
# 将该文字描述embedding
text_embedding = get_single_text_embedding(caption)
# 检索top_K相似图片
top_images = get_top_N_images_mix(query_image, caption, image_data_df)
plot_images_by_side(top_images)
