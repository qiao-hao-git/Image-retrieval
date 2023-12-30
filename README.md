# Image-retrieval

### 数据集
采用flickr30k数据集

### 1 使用Clip模型提取图像特征
[Clip模型下载链接](https://huggingface.co/openai/clip-vit-base-patch32)
### 2 使用Blip模型为图像生成文字描述
[Blip模型下载链接](https://huggingface.co/Salesforce/blip-image-captioning-base)
### 3 分别计算二者的相似度，检索top_K相似图像
