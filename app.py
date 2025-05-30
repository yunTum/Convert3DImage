import streamlit as st
import numpy as np
import torch
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import io
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import japanize_matplotlib  # 日本語フォント対応
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="画像から奥行き推定", layout="wide")

# 日本語フォント設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

@st.cache_resource
def load_model():
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    return feature_extractor, model

def predict_depth(image, feature_extractor, model):
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # 可視化のための処理
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    
    output = prediction.squeeze().cpu().numpy()
    
    # 原型の深度値を保存
    raw_depth = output.copy()
    
    # 正規化
    output = (output - output.min()) / (output.max() - output.min())
    return output, raw_depth

def estimate_distance(depth_map, known_center_distance=None, invert=True):
    """
    深度マップから距離を推定する
    
    Parameters:
    depth_map (numpy.ndarray): 深度マップ
    known_center_distance (float, optional): 中心点の既知の距離（メートル）
    invert (bool): 深度値を反転するかどうか（DPTモデルの深度値が実際の距離と逆の場合）
    
    Returns:
    tuple: (最小距離, 最大距離), スケール係数
    """
    # 中心点の深度値
    h, w = depth_map.shape
    center_depth = depth_map[h // 2, w // 2]
    
    if invert:
        # 深度値を反転: 大きな値ほど近く、小さな値ほど遠くなるようにする
        # 注意: DPTは大きな値ほど遠いという想定で出力するが、実際の配置と逆の場合があるため
        depth_map_processed = 1.0 / (depth_map + 1e-6)  # ゼロ除算を防ぐ
        center_depth_processed = 1.0 / (center_depth + 1e-6)
    else:
        depth_map_processed = depth_map
        center_depth_processed = center_depth
    
    if known_center_distance is not None and known_center_distance > 0:
        # 既知の中心点距離を使用してスケール係数を計算
        scale_factor = known_center_distance / center_depth_processed
    else:
        # デフォルトのスケール係数（推定値）
        scale_factor = 10.0
    
    min_processed = depth_map_processed.min()
    max_processed = depth_map_processed.max()
    
    # 最小距離と最大距離（メートル）の推定
    min_distance = min_processed * scale_factor
    max_distance = max_processed * scale_factor
    
    return (min_distance, max_distance), scale_factor, depth_map_processed

def create_depth_map_image(depth_map, colormap='plasma'):
    # カラーマップでの深度マップの可視化
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_map, cmap=colormap)
    plt.colorbar(label='正規化された深度')
    plt.title('深度マップ')
    plt.axis('off')
    
    # 一時ファイルとしてではなく、メモリ上に保存
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    
    # PILイメージとして読み込み
    depth_colored = Image.open(buf)
    return depth_colored

def create_raw_depth_image(depth_map):
    # 深度マップをグレースケールの画像に変換（生データ）
    # 正規化して0-255の範囲に収める
    normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_img = (normalized_depth * 255).astype(np.uint8)
    
    # PILイメージに変換
    depth_image = Image.fromarray(depth_img)
    return depth_image

def main():
    st.title("画像から奥行き（深度）推定")
    
    st.write("画像をアップロードして、その奥行き（深度）マップを生成します。")
    
    # サイドバーに設定項目を追加
    st.sidebar.header("設定")
    known_center_distance = st.sidebar.number_input(
        "中心点の距離（メートル）",
        min_value=0.1,
        max_value=100.0,
        value=None,
        step=0.1,
        help="中心点の既知の距離を入力すると、距離推定の精度が向上します。不明な場合は空欄にしてください。"
    )
    
    invert_depth = st.sidebar.checkbox(
        "深度値を反転する", 
        value=False,
        help="オンにすると、深度値を反転します（大きい値=近い、小さい値=遠い）。画像によって適切な設定が異なります。"
    )
    
    colormap_option = st.sidebar.selectbox(
        "カラーマップ", 
        options=["plasma", "viridis", "inferno", "magma", "cividis", "turbo"],
        index=0,
        help="深度マップの表示に使用するカラースキーム"
    )
    
    uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 画像を読み込む
        image = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner("奥行きを計算中..."):
            # モデルをロード
            feature_extractor, model = load_model()
            
            # 奥行き推定
            depth_map, raw_depth = predict_depth(image, feature_extractor, model)
            
            # 距離の推定
            (min_distance, max_distance), scale_factor, processed_depth = estimate_distance(
                raw_depth, known_center_distance, invert=invert_depth
            )
            
            # 深度マップの可視化画像を作成（グラフ用）- 処理後の深度マップを使用
            depth_colored = create_depth_map_image(processed_depth, colormap=colormap_option)
            
            # 生の深度画像を作成（比較表示用）- 処理後の深度マップを使用
            raw_depth_image = create_raw_depth_image(processed_depth)
            
            # ヒートマップ用の深度データを生成
            h, w = processed_depth.shape
            y_center = h // 2
            x_center = w // 2
            
            # 中心点、左上、右上、左下、右下の深度値
            center_depth = processed_depth[y_center, x_center]
            top_left = processed_depth[h//4, w//4]
            top_right = processed_depth[h//4, 3*w//4]
            bottom_left = processed_depth[3*h//4, w//4]
            bottom_right = processed_depth[3*h//4, 3*w//4]
            
            # 各点の推定距離（メートル）
            center_distance = center_depth * scale_factor
            tl_distance = top_left * scale_factor
            tr_distance = top_right * scale_factor
            bl_distance = bottom_left * scale_factor
            br_distance = bottom_right * scale_factor
        
        # レイアウト：変換前後の画像を横並びに表示
        st.subheader("変換前後の比較")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**元の画像**")
            st.image(image, use_column_width=True)
        
        with col2:
            st.write("**深度マップ（生画像）**")
            st.image(raw_depth_image, use_column_width=True)
            st.caption("明るい部分ほど近く、暗い部分ほど遠い" if invert_depth else "暗い部分ほど近く、明るい部分ほど遠い")
        
        # 下部に詳細情報を表示
        st.subheader("詳細情報")
        
        # タブで情報を整理
        tab1, tab2 = st.tabs(["推定距離情報", "グラフデータ"])
        
        with tab1:
            # キャリブレーション情報を表示
            if known_center_distance:
                st.success(f"中心点の距離 {known_center_distance:.2f}m をキャリブレーション基準として使用しています。")
            else:
                st.warning("中心点の距離が指定されていないため、デフォルトのスケール係数を使用しています。より正確な距離推定のために、サイドバーで中心点の距離を設定してください。")
            
            st.write(f"**推定距離範囲:** {min_distance:.2f}m ~ {max_distance:.2f}m")
            st.write(f"**使用スケール係数:** {scale_factor:.4f}")
            st.write(f"**深度反転:** {'あり' if invert_depth else 'なし'}")
            
            st.write("**画像内の位置別推定距離:**")
            
            # 位置別の距離情報を表形式で表示
            distance_data = {
                "位置": ["左上", "中央", "右上", "左下", "右下"],
                "推定距離 (m)": [f"{tl_distance:.2f}", f"{center_distance:.2f}", f"{tr_distance:.2f}", 
                              f"{bl_distance:.2f}", f"{br_distance:.2f}"]
            }
            st.table(distance_data)
            
            # 中心点距離の設定を促すメッセージ
            if not known_center_distance:
                st.info("ヒント: 撮影対象までの距離が分かっている場合は、サイドバーで中心点の距離を設定すると、より正確な距離推定ができます。")
            
            st.info("注意: 推定距離はモデルの出力に基づく近似値です。実際の距離とは異なる場合があります。")
        
        with tab2:
            # カラー深度マップの表示
            st.write("**深度マップ（カラー表示）**")
            st.image(depth_colored, use_column_width=True)
            
            # 深度の3Dサーフェスプロット
            st.write("**3D深度サーフェス**")
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # データが大きすぎる場合はダウンサンプリング
            downsample = 5
            x = np.arange(0, w, downsample)
            y = np.arange(0, h, downsample)
            # 反転したY座標で作成（上が0、下がh-1の画像座標系に合わせる）
            y_reversed = np.arange(h-1, -1, -downsample)
            X, Y = np.meshgrid(x, y_reversed)
            Z = processed_depth[::downsample, ::downsample]
            # Y軸の向きを反転
            Z = Z[::-1, :]
            
            surf = ax.plot_surface(X, Y, Z, cmap=colormap_option, linewidth=0, antialiased=False)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('深度')
            ax.set_title('3D深度サーフェス')
            # 視点を調整
            ax.view_init(elev=30, azim=225)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='深度')
            st.pyplot(fig)
            
            # Plotlyを使用したインタラクティブな3Dサーフェス
            st.write("**インタラクティブ3D深度サーフェス**")
            
            # ダウンサンプリング（さらに軽量化）
            downsample_interactive = 10  # より大きなダウンサンプリング係数
            x_interactive = np.arange(0, w, downsample_interactive)
            y_interactive = np.arange(0, h-1, -downsample_interactive)  # Y軸を反転
            Z_interactive = processed_depth[::downsample_interactive, ::downsample_interactive]
            # Y軸の向きを反転
            Z_interactive = Z_interactive[::-1, :]
            
            # インタラクティブ3Dサーフェスを作成
            fig_interactive = go.Figure(data=[
                go.Surface(
                    z=Z_interactive,
                    colorscale=colormap_option,
                    colorbar=dict(title="深度値")
                )
            ])
            
            fig_interactive.update_layout(
                title='インタラクティブ3D深度マップ（回転・ズーム可能）',
                scene=dict(
                    xaxis_title='X位置',
                    yaxis_title='Y位置',
                    zaxis_title='深度値',
                    # カメラ位置を調整
                    camera=dict(
                        eye=dict(x=-1.5, y=-1.5, z=1.0)
                    )
                ),
                width=800,
                height=800,
                margin=dict(l=0, r=0, b=0, t=40)
            )
            
            st.plotly_chart(fig_interactive, use_container_width=True)
            
            # 深度分布のヒストグラム
            st.write("**深度値の分布**")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(processed_depth.flatten(), bins=50, alpha=0.7, color='skyblue')
            ax.set_xlabel('深度値')
            ax.set_ylabel('頻度')
            ax.set_title('深度値の分布')
            st.pyplot(fig)
            
            # インタラクティブなプロファイルグラフ
            st.write("**インタラクティブ深度プロファイル**")
            
            # 画像上の位置選択
            profile_type = st.radio(
                "プロファイルの種類を選択",
                ["水平断面", "垂直断面", "任意の点の深度"]
            )
            
            if profile_type == "水平断面":
                # 水平断面の場合、Y座標を選択
                y_position = st.slider("Y位置（上から下）", 0, h-1, h//2)
                
                # プロファイルデータの取得
                profile_data = processed_depth[y_position, :]
                
                # Plotlyでインタラクティブグラフを作成
                fig_profile = px.line(
                    x=np.arange(w), 
                    y=profile_data,
                    labels={'x': 'X位置', 'y': '深度値'},
                    title=f'Y={y_position}における水平深度プロファイル'
                )
                fig_profile.update_layout(
                    hovermode="x",
                    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
                )
                
                # 画像上に水平線を表示
                fig_marker = plt.figure(figsize=(6, 6))
                plt.imshow(processed_depth, cmap=colormap_option)
                plt.axhline(y=y_position, color='red', linestyle='-', linewidth=2)
                plt.title(f'Y={y_position}の位置')
                plt.axis('off')
                
                # 2列に分けて表示
                col_profile1, col_profile2 = st.columns(2)
                with col_profile1:
                    st.pyplot(fig_marker)
                with col_profile2:
                    st.plotly_chart(fig_profile, use_container_width=True)
                
            elif profile_type == "垂直断面":
                # 垂直断面の場合、X座標を選択
                x_position = st.slider("X位置（左から右）", 0, w-1, w//2)
                
                # プロファイルデータの取得
                profile_data = processed_depth[:, x_position]
                
                # Plotlyでインタラクティブグラフを作成
                fig_profile = px.line(
                    x=np.arange(h), 
                    y=profile_data,
                    labels={'x': 'Y位置', 'y': '深度値'},
                    title=f'X={x_position}における垂直深度プロファイル'
                )
                fig_profile.update_layout(
                    hovermode="x",
                    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
                )
                
                # 画像上に垂直線を表示
                fig_marker = plt.figure(figsize=(6, 6))
                plt.imshow(processed_depth, cmap=colormap_option)
                plt.axvline(x=x_position, color='red', linestyle='-', linewidth=2)
                plt.title(f'X={x_position}の位置')
                plt.axis('off')
                
                # 2列に分けて表示
                col_profile1, col_profile2 = st.columns(2)
                with col_profile1:
                    st.pyplot(fig_marker)
                with col_profile2:
                    st.plotly_chart(fig_profile, use_container_width=True)
                
            else:  # 任意の点の深度
                st.write("画像上の任意の点の深度値を調べます")
                
                col_point1, col_point2 = st.columns(2)
                with col_point1:
                    x_point = st.slider("X位置（左から右）", 0, w-1, w//2)
                with col_point2:
                    y_point = st.slider("Y位置（上から下）", 0, h-1, h//2)
                
                # 選択した点の深度値
                point_depth = processed_depth[y_point, x_point]
                point_distance = point_depth * scale_factor
                
                # 半径10ピクセル内の深度値を抽出
                radius = 10
                y_min = max(0, y_point - radius)
                y_max = min(h, y_point + radius + 1)
                x_min = max(0, x_point - radius)
                x_max = min(w, x_point + radius + 1)
                
                neighborhood = processed_depth[y_min:y_max, x_min:x_max]
                mean_depth = np.mean(neighborhood)
                std_depth = np.std(neighborhood)
                
                # 画像上に点を表示
                fig_point = plt.figure(figsize=(6, 6))
                plt.imshow(processed_depth, cmap=colormap_option)
                plt.scatter(x_point, y_point, color='red', s=100, marker='x')
                circle = plt.Circle((x_point, y_point), radius, color='red', fill=False, linestyle='--')
                plt.gca().add_patch(circle)
                plt.title(f'選択した点 (X={x_point}, Y={y_point})')
                plt.axis('off')
                
                # 3Dヒートマップ（周辺領域）
                z_data = neighborhood
                x_surface = np.arange(x_min, x_max)
                y_surface = np.arange(y_min, y_max)
                X_surface, Y_surface = np.meshgrid(x_surface, y_surface)
                
                fig_surface = go.Figure(data=[
                    go.Surface(
                        z=z_data,
                        x=X_surface,
                        y=Y_surface,
                        colorscale=colormap_option,
                        colorbar=dict(title="深度値")
                    )
                ])
                fig_surface.update_layout(
                    title=f'選択点周辺の深度サーフェス',
                    scene=dict(
                        xaxis_title='X位置',
                        yaxis_title='Y位置',
                        zaxis_title='深度値',
                        camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))
                    ),
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                
                # 情報表示
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.pyplot(fig_point)
                    st.write(f"**選択点の深度値:** {point_depth:.4f}")
                    st.write(f"**推定距離:** {point_distance:.2f}m")
                with col_info2:
                    st.plotly_chart(fig_surface, use_container_width=True)
                    st.write(f"**周辺領域の平均深度値:** {mean_depth:.4f} (標準偏差: {std_depth:.4f})")
                    st.write(f"**周辺領域の推定距離:** {mean_depth * scale_factor:.2f}m")
        
        # ダウンロードボタン
        st.subheader("ダウンロード")
        col_a, col_b = st.columns(2)
        
        with col_a:
            # 生深度マップのダウンロード
            buf_raw = io.BytesIO()
            raw_depth_image.save(buf_raw, format="PNG")
            byte_raw = buf_raw.getvalue()
            st.download_button(
                label="生深度マップをダウンロード",
                data=byte_raw,
                file_name="raw_depth_map.png",
                mime="image/png"
            )
        
        with col_b:
            # カラー深度マップのダウンロード
            buf_color = io.BytesIO()
            depth_colored.save(buf_color, format="PNG")
            byte_color = buf_color.getvalue()
            st.download_button(
                label="カラー深度マップをダウンロード",
                data=byte_color,
                file_name="color_depth_map.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main() 