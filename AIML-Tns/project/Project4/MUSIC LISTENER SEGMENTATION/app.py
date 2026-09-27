import gradio as gr
import pandas as pd
import joblib

from fastapi import FastAPI
from gradio.routes import mount_gradio_app

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def get_segment(cluster):

    centres = scaler.inverse_transform(
        model.cluster_centers_
    )

    activity_scores = []

    for centre in centres:

        listening_hours = centre[0]
        songs_per_day = centre[1]
        skip_rate = centre[2]
        playlist_count = centre[3]

        score = (
            listening_hours
            + songs_per_day
            + playlist_count
            - skip_rate
        )

        activity_scores.append(score)

    sorted_clusters = sorted(
        range(len(activity_scores)),
        key=lambda x: activity_scores[x]
    )

    casual_cluster = sorted_clusters[0]
    explorer_cluster = sorted_clusters[1]
    heavy_cluster = sorted_clusters[2]

    if cluster == casual_cluster:
        return "🎧 Casual Listener"

    elif cluster == explorer_cluster:
        return "🎵 Music Explorer"

    elif cluster == heavy_cluster:
        return "🔥 Heavy Listener"

    return "Unknown"

def predict_listener(
    listening_hours,
    songs_per_day,
    skip_rate,
    playlist_count
):

    user_data = pd.DataFrame([{

        "listening_hours_per_week":
            listening_hours,

        "songs_per_day":
            songs_per_day,

        "skip_rate":
            skip_rate,

        "playlist_count":
            playlist_count

    }])

    user_data_scaled = scaler.transform(
        user_data
    )

    cluster = int(
        model.predict(
            user_data_scaled
        )[0]
    )

    segment = get_segment(cluster)

    centres = scaler.inverse_transform(
        model.cluster_centers_
    )

    centre = centres[cluster]

    result = f"""

**K-Means Cluster:** {cluster}

---

| Feature | Value |
|---|---:|
| Listening Hours/Week | {listening_hours:.1f} |
| Songs/Day | {songs_per_day:.1f} |
| Skip Rate | {skip_rate:.1f}% |
| Playlist Count | {playlist_count} |

---

| Feature | Cluster Average |
|---|---:|
| Listening Hours/Week | {centre[0]:.2f} |
| Songs/Day | {centre[1]:.2f} |
| Skip Rate | {centre[2]:.2f}% |
| Playlist Count | {centre[3]:.2f} |

"""

    return result

app = FastAPI(
    title="Music Listener Segmentation API",
    description="K-Means based music listener segmentation",
    version="1.0"
)

with gr.Blocks(
    title="Music Listener Segmentation"
) as demo:

    gr.Markdown(
        """

Enter the listener's behaviour using the interactive
controls below to identify their listener segment.
"""
    )

    gr.Markdown(
        "## 🎧 Listener Behaviour"
    )

    with gr.Row():

        listening_hours = gr.Slider(
            minimum=0,
            maximum=168,
            value=10,
            step=0.5,
            label="Listening Hours per Week"
        )

        songs_per_day = gr.Slider(
            minimum=0,
            maximum=150,
            value=20,
            step=1,
            label="Songs per Day"
        )

    with gr.Row():

        skip_rate = gr.Slider(
            minimum=0,
            maximum=100,
            value=25,
            step=1,
            label="Skip Rate (%)"
        )

        playlist_count = gr.Slider(
            minimum=0,
            maximum=100,
            value=5,
            step=1,
            label="Playlist Count"
        )

    analyze_button = gr.Button(
        "🔍 Analyze Listener",
        variant="primary"
    )

    result = gr.Markdown(
        value="""

Move the sliders and click **Analyze Listener**.
"""
    )

    analyze_button.click(

        fn=predict_listener,

        inputs=[
            listening_hours,
            songs_per_day,
            skip_rate,
            playlist_count
        ],

        outputs=result
    )

    gr.Markdown(
        """
---

The K-Means model uses four behavioural features:

- **Listening Hours per Week**
- **Songs per Day**
- **Skip Rate**
- **Playlist Count**

The model automatically groups listeners into:

- 🎧 Casual Listener
- 🎵 Music Explorer
- 🔥 Heavy Listener

The labels are assigned **after analysing the K-Means
cluster centres**. K-Means itself only produces cluster
numbers.
"""
    )

app = mount_gradio_app(
    app,
    demo,
    path="/"
)