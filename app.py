import os
import sys

# Set environment variable to disable Streamlit file watcher for torch modules
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import cv2
import numpy as np

# Page config
st.set_page_config(
    page_title="People Counter",
    page_icon="👥",
    layout="wide"
)

# Lazy import functions
@st.cache_resource
def load_video_processor():
    """Lazy load video processing components"""
    try:
        from models.model import process_video, model
        return process_video, model
    except Exception as e:
        st.error(f"Error loading video processor: {e}")
        return None, None

@st.cache_resource  
def load_live_camera():
    """Lazy load live camera components"""
    try:
        from camera import run_live_counter
        return run_live_counter
    except Exception as e:
        st.error(f"Error loading live camera: {e}")
        return None

@st.cache_resource
def load_face_recognition():
    """Lazy load face recognition components"""
    try:
        from face_recognition_system import run_face_recognition_system
        return run_face_recognition_system
    except Exception as e:
        st.error(f"Error loading face recognition: {e}")
        return None

@st.cache_resource
def load_integrated_tracking():
    """Lazy load integrated tracking components"""
    try:
        from integrated_tracking import run_integrated_people_tracking
        return run_integrated_people_tracking
    except Exception as e:
        st.error(f"Error loading integrated tracking: {e}")
        return None

# Main app
st.title("👥 People Counting System")
st.markdown("Advanced people counting using YOLO - Support for videos and live cameras")

# Sidebar for mode selection
st.sidebar.title("Select Mode")
mode = st.sidebar.radio(
    "Choose mode:",
    ["📹 Video File Processing", "🎥 Live Camera Counting", "👤 Face Recognition System", "🎯 Integrated Tracking + Face Recognition"]
)

if mode == "📹 Video File Processing":
    st.header("📹 Advanced Video People Counting")
    st.markdown("Upload a video to count unique people with 5-second accuracy delay")
    
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

    if video_file is not None:
        # Save uploaded file to a temporary location
        video_path = 'data/uploaded_video.mp4'
        with open(video_path, 'wb') as f:
            f.write(video_file.read())
        
        output_path = 'results/output_video.mp4'
        
        # Processing settings
        col1, col2 = st.columns(2)
        with col1:
            confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.1)
        with col2:
            st.info("📊 5-second delay for accurate counting")
        
        if st.button("🎬 Process Video & Count People", type="primary"):
            # Load video processor
            process_video, model = load_video_processor()
            
            if process_video is not None and model is not None:
                # Create progress containers
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_container = st.empty()
                
                def update_progress(progress, stats):
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {stats['current_frame']}/{stats['total_frames']} - Current: {stats['current_count']} people")
                    
                    if stats['unique_count'] > 0:
                        with metrics_container.container():
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Current Frame", stats['current_count'])
                            col2.metric("Max in Frame", stats['max_count'])
                            col3.metric("Unique People", stats['unique_count'])
                
                with st.spinner('🎥 Processing video with people counting...'):
                    try:
                        results = process_video(
                            video_path, 
                            output_path, 
                            model, 
                            confidence=confidence,
                            progress_callback=update_progress
                        )
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Video processing completed!")
                        
                        # Display final results
                        st.success('🎉 Video processed successfully!')
                        
                        # Results summary
                        st.subheader("📊 Counting Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(
                                "🚶‍♂️ Unique People", 
                                results['unique_people_count'],
                                help="Total unique individuals detected (with 5s delay for accuracy)"
                            )
                        with col2:
                            st.metric(
                                "👥 Max in Frame", 
                                results['max_people_in_frame'],
                                help="Maximum people visible simultaneously"
                            )
                        with col3:
                            st.metric(
                                "📈 Average per Frame", 
                                f"{results['average_people_per_frame']:.1f}",
                                help="Average people count across all frames"
                            )
                        with col4:
                            st.metric(
                                "⏱️ Video Duration", 
                                f"{results['video_duration_seconds']:.1f}s",
                                help="Total video length processed"
                            )
                        
                        # Additional details
                        with st.expander("📋 Detailed Statistics"):
                            st.write(f"**Total frames processed:** {results['total_frames_processed']:,}")
                            st.write(f"**Total detections:** {results['total_detections']:,}")
                            st.write(f"**Detection confidence:** {results['confidence_threshold']}")
                            st.write(f"**Counting delay:** {results['detection_delay_seconds']} seconds")
                            st.write(f"**Processing method:** Advanced centroid tracking with temporal filtering")
                        
                        # Display processed video
                        st.subheader("🎥 Processed Video")
                        st.video(results['output_path'])
                        
                        # Download link
                        with open(results['output_path'], 'rb') as file:
                            st.download_button(
                                label="📥 Download Processed Video",
                                data=file.read(),
                                file_name=f"people_counted_video_{results['unique_people_count']}_people.mp4",
                                mime="video/mp4",
                                help="Download the video with people detection overlays"
                            )
                            
                    except Exception as e:
                        st.error(f"❌ Error processing video: {str(e)}")
                        st.info("💡 Try reducing the video size or using a different format")
            else:
                st.error("❌ Failed to load video processing components. Please check your installation.")
                st.info("💡 Try restarting the application or check the model files.")

elif mode == "🎥 Live Camera Counting":
    # Load live camera function
    run_live_counter = load_live_camera()
    
    if run_live_counter is not None:
        run_live_counter()
    else:
        st.error("Failed to load live camera components. Please check your installation.")
        st.info("Try using the simplified live camera app: `streamlit run live_test.py`")

elif mode == "👤 Face Recognition System":
    # Load face recognition function
    run_face_recognition_system = load_face_recognition()
    
    if run_face_recognition_system is not None:
        run_face_recognition_system()
    else:
        st.error("❌ Failed to load face recognition components.")
        st.info("💡 The system can still work in OpenCV fallback mode.")
        
        with st.expander("🔧 Installation Help"):
            st.markdown("""
            **Quick Fix Options:**
            
            1. **Install face_recognition** (recommended):
               ```bash
               pip install cmake
               pip install dlib  
               pip install face_recognition
               ```
            
            2. **Use OpenCV fallback mode**: The app works with basic face detection
            
            3. **Skip face recognition**: Use the other modes (Video/Live Camera) - they work perfectly!
            
            **📖 See `INSTALLATION_HELP.md` for detailed Windows installation guide**
            """)
        
        # Offer alternative modes
        st.markdown("### 🎯 **Try These Working Modes:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📹 Video Processing", type="primary"):
                st.rerun()
        with col2:
            if st.button("🎥 Live Camera", type="primary"):
                st.rerun()

elif mode == "🎯 Integrated Tracking + Face Recognition":
    # Load integrated tracking function
    run_integrated_people_tracking = load_integrated_tracking()
    
    if run_integrated_people_tracking is not None:
        # Check if face recognition is available
        try:
            import face_recognition
            run_integrated_people_tracking()
        except ImportError:
            st.warning("⚠️ **Integrated Tracking - Limited Mode**")
            st.info("💡 Running with YOLO people detection only (no face recognition)")
            st.markdown("**Install face_recognition for full functionality**: `pip install face_recognition`")
            
            # Still offer the basic live camera mode
            st.markdown("### 🎥 **Use Live Camera Mode Instead:**")
            if st.button("🎥 Switch to Live Camera", type="primary"):
                st.session_state.mode = "🎥 Live Camera Counting"
                st.rerun()
    else:
        st.error("❌ Failed to load integrated tracking components.")
        st.info("💡 Try the Live Camera mode for people counting")
