# キカガク　DXを推進するAI・データサイエンス人材育成コース　給付金要件　自走期間課題の提出
# 設定課題 XAI

# 数字文字の分類を行うモジュール

# 必要なモジュールをインポートする
# import streamlit as st    # Comment out on 2024/01/03
# shapモジュール、datasetの読み込みに時間がかかるので、見える安心感を作るためにプログレスバーを表示
# progress_text = "Operation in progress. Please wait." # Comment out on 2024/01/03
# my_bar = st.progress(0.1, text=progress_text) # Comment out on 2024/01/03

# 必要なモジュールをインポートする
import shap
import numpy as np

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets

import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import japanize_matplotlib

# 自作モジュールにアクセスできるようにする
# import my1_cvtmnist as my_cvtmnist
import my2_cnn as my_cnn
# import my3_predictX as my_predict


# # プログレスバーを適切に伸ばす
# my_bar.progress(0.5, text=progress_text)    # Comment out on 2024/01/03


batch_size = 128
num_epochs = 2
device = torch.device('cpu')

# shapモジュール用とテスト画像用のためのデータセットをロード
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('mnist_data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor()
#                    ])),
#     batch_size=batch_size, shuffle=True)

# # プログレスバーを適切に伸ばす
# my_bar.progress(0.85, text=progress_text)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./', train=False, download=True, 
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=batch_size, shuffle=True)


# # プログレスバーを一瞬100％にして瞬消す
# my_bar.progress(1.0, text=progress_text)
# my_bar.empty()


# 画像分類で数字を戻す
def PredictNumber(im):
    model = my_cnn.Net1().eval()
    # 重みの読み込み
    model.load_state_dict(torch.load('./mnist_pl.pt', map_location=torch.device('cpu')))

    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(im)

    img_tensor = img_tensor.unsqueeze(0)

    # 予測の実行
    y = model(img_tensor)

    # 確率に変換
    y = F.softmax(y, dim=1)

    # 予測ラベル
    y = torch.argmax(y)

    return y


# 画像の特徴量ヒートマップをローカルに保存する
def eXplainableAI(im):
    model = my_cnn.Net2().eval()
    # 重みの読み込み
    model.load_state_dict(torch.load('./mnist_nn.pt', map_location=torch.device('cpu')))


    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(im)

    img_tensor = img_tensor.unsqueeze(0)


    # since shuffle=True, this is a random sample of test data
    batch = next(iter(test_loader))
    # images, _ = batch
    images, images_nums = batch


    background = images[:100]


    test_images = images[100:101]
    test_images_nums = images_nums[100:102]


    # 予測の実行
    y = model(img_tensor)
    print('Model 予測結果(Uploaded data)', y)

    # 確率に変換
    y = F.softmax(y, dim=1)
    print('　確率：', y)

    # 予測ラベル
    y = torch.argmax(y, dim=1)
    print('　予測ラベル：', y)

    # 予測の実行
    y = model(test_images[0])
    print('Model 予測結果(Test data)', y)

    # 確率に変換
    y = F.softmax(y, dim=1)
    print('　確率：', y)

    # 予測ラベル
    y = torch.argmax(y)
    print('　予測ラベル：', y)
    print('--')


    # テンソルを行方向に連結
    result_tensor = torch.cat((test_images, img_tensor), dim=0)
    test_images = result_tensor
    test_images_nums[1] = y


    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test_images)

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

    # plot the feature attributions
    shap.image_plot(shap_numpy, -test_numpy, show=False)
    # plt.suptitle('shap 手描き画像をテスト')
    plt.savefig('./temp_img/shap_plot white.png')
    # plt.show()

    return 
