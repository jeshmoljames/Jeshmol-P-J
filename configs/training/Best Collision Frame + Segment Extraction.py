def get_best_collision_and_segments(video_path, fps=5):
    video = VideoFileClip(video_path)

    best_frame = None
    best_score = -1

    segments = []
    current_segment = None

    frame_idx = 0

    for frame in video.iter_frames(fps=fps):
        objects = detect_objects(frame)
        collisions = detect_collisions(objects)

        timestamp = frame_idx / fps

        if collisions:
            max_score = max([c[2] for c in collisions])

            # Best frame selection
            if max_score > best_score:
                best_score = max_score
                best_frame = frame.copy()

            # Segment tracking
            if current_segment is None:
                current_segment = [timestamp, timestamp]
            else:
                current_segment[1] = timestamp

        else:
            if current_segment is not None:
                segments.append(tuple(current_segment))
                current_segment = None

        frame_idx += 1

    # Handle last segment
    if current_segment is not None:
        segments.append(tuple(current_segment))

    return best_frame, best_score, segments
