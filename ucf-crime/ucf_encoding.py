# General
import numpy as np
import argparse
import os
from alive_progress import alive_bar
import cv2
import time

# My code
from my_utils import read_uca_as_df
from vision_encoders import possible_models, EncoderBuilder
from databases import possible_databases, DatabaseBuilder


CLIP_DURATION_S = 14
def ucaless_encode(ucf_path, save_path, encoder, database):

    # Build all necessary
    model = EncoderBuilder.build(encoder_name=encoder)

    database_handler = DatabaseBuilder.build(database_name=database, encoder_params=model.get_encoder_params())

    # List all videos
    ucf_videos_list = [os.path.join(ucf_path, video_file) for video_file in os.listdir(ucf_path)]
    
    with alive_bar(len(ucf_videos_list)):
        # Go through each UCF video
        for video in ucf_videos_list:

            print(f"[MAIN]: Encoding video {video.split('/')[-1]}")

            # Initialize video capture
            cap = cv2.VideoCapture(video)

            # Get FPS
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Clip duration in frames
            max_frames = int(CLIP_DURATION_S * fps)

            # Read first frame
            ret, frame = cv2.read()

            start_frame = 0
            end_frame = -1
            i = 0
            # Read all frames
            while ret:
                
                i+=1
                start_frame = end_frame + 1
                # Read a clip
                clip_frames = []
                for frame_index in (range(max_frames)):
                    if not ret:
                        frame_index -= 1
                        break
                    
                    clip_frames.append(frame)

                    # Read next
                    ret, frame = cv2.read()
                
                # Obtain metadata
                end_frame = frame_index

                if end_frame != -1:

                    print(f'\tEncoding clip number {i}')
                    # Encode model
                    #emb = model.get_clip_embedding(clip_frames)

                    # Save file
                    if save_path:
                        np_file_name = video.split('.')[0] + '_' + str(start_frame) + '-' + str(end_frame) + '.npy'
                        #np.save(os.path.join(save_path, np_file_name), emb)

                    print(f'\tUploading clip number {i}')
                    # Upload to database
                    # database_handler.upload_embedding(id,
                    #                                 emb,
                    #                                 metadata={'video':video.split('.')[0],
                    #                                             'start_frame':start_frame,
                    #                                             'end_frame':end_frame})
            
            # Release video capture
            cap.release()


def uca_encode(ucf_path, uca_path, save_path, encoder, database):

    # Read annotations dataset
    uca_df = read_uca_as_df(uca_path=uca_path)

    # Load model
    model = EncoderBuilder.build(encoder_name=encoder)

    # Connect to database
    database_handler = DatabaseBuilder.build(database_name=database,
                                            encoder_params=model.get_encoder_params())

    # Get total number of annotations
    total = len(uca_df['timestamp'].tolist())

    id = 0
    # Create embeddings
    with alive_bar(int(total)) as bar:
        for root, dirs, files in os.walk(ucf_path):
            if not dirs: # It is a category's folder
                for video in files:

                    print(f"[MAIN]: Encoding video {video}")
                    
                    # Get all annotations belonging to this video
                    sub_df = uca_df[uca_df['video'] == video.split('.')[0]]

                    # Read video
                    video_path = os.path.join(root, video)
                    cap = cv2.VideoCapture(video_path)

                    # Read one frame to check video was properly read
                    ret, frame = cap.read()
                    if ret:

                        # Get FPS
                        fps = cap.get(cv2.CAP_PROP_FPS)

                        # Sort by timestamps
                        sub_df = sub_df.sort_values(by='timestamp', key=lambda x: x.apply(lambda y: y[0]))

                        # Get timestamps
                        #timestamps = sub_df['timestamp'].tolist()
                        #sorted_timestamps = sorted(timestamps, key=lambda x: x[0])

                        for entry in sub_df.itertuples():
                            
                            sentence = entry.sentence
                            dataset = entry.dataset
                            class_name = entry.class_name
                            timestamp = entry.timestamp
                            
                            print(f"[MAIN]: Encoding clip with timestamps {timestamp}")
                            # Get start and end frame
                            start_frame, end_frame = int(timestamp[0]*fps), int(timestamp[1]*fps)
                            
                            # Go to start frame in video
                            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                            # Read frames
                            clip_frames = []
                            for i in range(start_frame, end_frame, int(fps)):
                                ret, frame = cap.read()
                                if ret:
                                    clip_frames.append(frame)

                            # Get embedding of clip
                            start_emb = time.perf_counter()
                            emb = model.get_clip_embedding(clip_frames)
                            end_emb = time.perf_counter()
                            emb_time = end_emb - start_emb

                            # Save embedding to file (if specified)
                            if save_path:
                                np_file_name = video.split('.')[0] + '_' + str(start_frame) + '-' + str(end_frame) + '.npy'
                                np.save(os.path.join(save_path, np_file_name), emb)

                            # Upload embedding to database
                            start_upload = time.perf_counter()
                            
                            database_handler.upload_embedding(id,
                                                            emb,
                                                            metadata={'video':video.split('.')[0],
                                                                        'start_frame':start_frame,
                                                                        'end_frame':end_frame,
                                                                        'sentence':sentence,
                                                                        'dataset':dataset,
                                                                        'class_name':class_name})
                            
                            end_upload = time.perf_counter()
                            upload_time = end_upload - start_upload
                            id += 1

                            # Clip embedded
                            bar()
                    else:
                        print(f"[MAIN]: Video {video} could not be read")

                    cap.release()

def get_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--ucf-path', type=str, default='/media/pablo/358690d7-e500-45fb-b8f8-bc48c6be13e3/UCF-Crimes/Videos', help='Directory of the UCF Crime dataset')
    parser.add_argument('--just-ucf', type=bool, default=False)
    parser.add_argument('--uca-path', type=str, default='/media/pablo/358690d7-e500-45fb-b8f8-bc48c6be13e3/Surveillance-Video-Understanding/UCF Annotation/json', help='Directory of the UCA dataset\' JSON file')
    parser.add_argument('--save-path', type=str, help='Directory where embeddings of the clips will be saved. If not specified, embeddings will not be stored locally')
    parser.add_argument('--encoder', type=str, choices=possible_models, required=True, help='The encoder used to generate the embeddings of the clips')
    parser.add_argument('--database', type=str, choices=possible_databases, required=True, help='The database used to store the embeddings of the clips')

    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()

    if args.just_ucf:
        ucaless_encode(args.ucf_path, args.save_path, args.encoder, args.database)
    else:
        uca_encode(args.ucf_path, args.uca_path, args.save_path, args.encoder, args.database)
