import cv2
import csv
import numpy as np
from collections import defaultdict


def load_csv(csv_path):
    """Load CSV file and organize events by frame."""
    events_by_frame = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            frame = int(row[0])
            activity = float(row[1])
            azimuth = float(row[2])
            elevation = float(row[3])
            cat1 = row[4]
            sim1 = float(row[5])
            cat2 = row[6]
            sim2 = float(row[7])
            cat3 = row[8]
            sim3 = float(row[9])
            events_by_frame[frame].append({
                'activity': activity,
                'azimuth': azimuth,
                'elevation': elevation,
                'categories': [
                    (cat1, sim1),
                    (cat2, sim2),
                    (cat3, sim3),
                ],
            })
    return events_by_frame


def azimuth_elevation_to_pixel(azimuth, elevation, width, height):
    """
    Convert azimuth and elevation to pixel coordinates in an equirectangular image.
    azimuth: -180 ~ 180 (degrees), -180 is left, 180 is right
    elevation: -90 ~ 90 (degrees), 90 is top, -90 is bottom
    """
    x = int((180.0 - azimuth) / 360.0 * width)
    y = int((90.0 - elevation) / 180.0 * height)
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    return x, y


def get_color_by_activity(activity):
    """Return a color (BGR) based on the activity value."""
    if activity > 0.8:
        return (0, 0, 255)      # Red: high activity
    elif activity > 0.5:
        return (0, 165, 255)    # Orange: medium activity
    else:
        return (0, 255, 255)    # Yellow: low activity


def draw_events_on_frame(frame_img, events, width, height):
    """Draw event markers and text information on a video frame."""
    overlay = frame_img.copy()

    for event in events:
        azimuth = event['azimuth']
        elevation = event['elevation']
        activity = event['activity']
        categories = event['categories']

        # Convert azimuth/elevation to pixel coordinates
        x, y = azimuth_elevation_to_pixel(azimuth, elevation, width, height)

        # Determine color based on activity
        color = get_color_by_activity(activity)

        # Marker size proportional to activity
        radius = int(15 + activity * 25)

        # Draw marker (circle with center dot)
        cv2.circle(overlay, (x, y), radius, color, 3)
        cv2.circle(overlay, (x, y), 5, color, -1)  # Center dot

        # Draw crosshair lines
        cross_size = radius + 5
        cv2.line(overlay, (x - cross_size, y), (x + cross_size, y), color, 1)
        cv2.line(overlay, (x, y - cross_size), (x, y + cross_size), color, 1)

        # Text rendering settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 20

        # Text position (upper right of the marker)
        text_x = x + radius + 10
        text_y = y - radius

        # If text exceeds right edge, display on the left side
        if text_x + 250 > width:
            text_x = x - radius - 260

        # If text exceeds top edge, display below the marker
        if text_y < 30:
            text_y = y + radius + 20

        # Draw background box for text
        box_width = 280
        box_height = line_height * 5 + 10
        cv2.rectangle(
            overlay,
            (text_x - 5, text_y - 15),
            (text_x + box_width, text_y + box_height),
            (0, 0, 0),
            -1
        )
        cv2.rectangle(
            overlay,
            (text_x - 5, text_y - 15),
            (text_x + box_width, text_y + box_height),
            color,
            2
        )

        # Display activity value
        act_text = f"Activity: {activity:.3f}"
        cv2.putText(overlay, act_text, (text_x, text_y),
                    font, font_scale, (255, 255, 255), thickness)

        # Display azimuth and elevation
        pos_text = f"Az: {azimuth:.1f} El: {elevation:.1f}"
        cv2.putText(overlay, pos_text, (text_x, text_y + line_height),
                    font, font_scale, (200, 200, 200), thickness)

        # Display top 3 categories with similarity scores
        for i, (cat, sim) in enumerate(categories):
            cat_text = f"{i+1}. {cat}: {sim:.3f}"
            cat_color = (0, 255, 0) if sim > 0 else (128, 128, 128)
            cv2.putText(overlay, cat_text,
                        (text_x, text_y + line_height * (i + 2)),
                        font, font_scale, cat_color, thickness)

    # Blend overlay with original frame for semi-transparency
    result = cv2.addWeighted(overlay, 0.9, frame_img, 0.1, 0)
    return result


def overlay_video(video_path, csv_path, output_path, frame_duration=0.1):
    """
    Main function: overlay CSV event results onto a 360 equirectangular video.
    frame_duration: duration of one CSV frame in seconds (default: 0.1s)
    """
    # Load CSV events
    events_by_frame = load_csv(csv_path)

    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height}, {video_fps}fps, {total_video_frames} frames")
    print(f"CSV events: {len(events_by_frame)} unique frames")

    # Set up output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

    video_frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate the corresponding CSV frame for the current video frame
        current_time = video_frame_idx / video_fps
        csv_frame = int(current_time / frame_duration)

        # Display frame info at the top of the video
        info_text = f"Frame: {video_frame_idx} | CSV Frame: {csv_frame} | Time: {current_time:.2f}s"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # Position with some margin
        text_x = 10                    # 10px from left
        text_y = 30                    # 30px from top

        cv2.putText(
            frame,
            info_text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

        # Draw events if they exist for the current CSV frame
        if csv_frame in events_by_frame:
            events = events_by_frame[csv_frame]
            frame = draw_events_on_frame(frame, events, width, height)

        out.write(frame)
        video_frame_idx += 1

        if video_frame_idx % 100 == 0:
            print(f"Processing: {video_frame_idx}/{total_video_frames} "
                  f"({video_frame_idx / total_video_frames * 100:.1f}%)")

    cap.release()
    out.release()
    print(f"Done! Output saved to {output_path}")


if __name__ == '__main__':
    video_path = './data_inference/example/fold3_room6_mix001_157to172sec.mp4'
    csv_path = './data_fsd50k_tau-srir/model_monitor/20251029062140_154940/inference_example_foa_0040000/fold3_room6_mix001_157to172sec.csv'
    output_path = './data_fsd50k_tau-srir/model_monitor/20251029062140_154940/inference_example_foa_0040000/fold3_room6_mix001_157to172sec_with_events.mp4'

    overlay_video(video_path, csv_path, output_path)
