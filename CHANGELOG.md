# Changelog

All notable changes to the People Counting System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-06-27

### 🎉 Major Release - Complete System Refactor

#### Added
- **Universal Camera Support** - USB, IP, RTSP, phone cameras via .env configuration
- **Unique Person Tracking** - Advanced centroid-based tracking prevents double counting
- **Face Recognition System** - Optional face identification with local database
- **Integrated Tracking** - Combined YOLO detection + face recognition
- **Comprehensive Testing Suite** - Multiple test scripts for system validation
- **Professional Documentation** - Complete guides for setup, camera config, and face recognition
- **Environment Configuration** - .env file support for flexible camera settings
- **Batch Scripts** - Easy Windows launchers (start_app.bat, run.bat, run.ps1)
- **Lazy Loading** - Improved performance with on-demand module loading
- **Error Handling** - Robust error management with user-friendly messages
- **Multi-angle Face Registration** - 5-point capture system for better accuracy
- **Data Augmentation** - Automatic face image variations for training
- **Privacy-First Design** - All processing done locally, no cloud dependencies
- **Real-time Analytics** - Live FPS monitoring and performance metrics
- **Snapshot Functionality** - Capture moments with timestamp and count data
- **Advanced UI** - Modern Streamlit interface with intuitive navigation

#### Enhanced
- **YOLO Model** - Upgraded to YOLOv9 for better accuracy and performance
- **Video Processing** - Improved unique counting algorithm with tracking persistence
- **Camera Handling** - Better error recovery and multi-camera support
- **Performance** - Optimized for real-time processing with configurable settings
- **Code Structure** - Modular design for easy maintenance and extension
- **Documentation** - Comprehensive README with clear setup instructions
- **Dependencies** - Updated to latest stable versions with clear separation of core/optional

#### Technical Improvements
- **Modular Architecture** - Separated concerns into logical modules
- **Configuration Management** - Environment-based settings for deployment flexibility
- **Testing Framework** - Comprehensive test coverage for all components
- **Error Recovery** - Graceful handling of camera disconnections and model failures
- **Resource Management** - Proper cleanup of camera resources and memory
- **Cross-platform Support** - Improved compatibility across Windows, macOS, and Linux

#### Security & Privacy
- **Local Processing** - No data sent to external servers
- **Secure Face Storage** - Encrypted local face database
- **Privacy Controls** - Clear data handling and storage policies
- **Access Control** - Configurable permissions for face recognition features

### [1.0.0] - Previous Version
- Basic people counting functionality
- Simple video upload processing
- Basic camera support
- Initial Streamlit interface

---

## 🚀 Future Roadmap

### Planned Features
- **Multi-zone Counting** - Define specific areas for counting
- **API Endpoints** - REST API for integration with other systems
- **Database Integration** - Support for external databases
- **Mobile App** - Companion mobile application
- **Cloud Deployment** - Docker containers and cloud-ready configurations
- **Advanced Analytics** - Heatmaps, flow analysis, and detailed reporting
- **Export Capabilities** - CSV, JSON, and PDF report generation
- **Real-time Alerts** - Notifications for capacity thresholds
- **Multi-camera Sync** - Coordinate multiple camera feeds
- **Edge Computing** - Optimized models for edge devices

### Technical Debt
- [ ] Add comprehensive unit tests
- [ ] Implement CI/CD pipeline
- [ ] Add performance benchmarking
- [ ] Create Docker deployment
- [ ] Add database abstraction layer

---

## 📝 Contribution Guidelines

When contributing to this project:

1. **Version Bumping**: Update VERSION file according to semantic versioning
2. **Changelog**: Add entries to this file following the format above
3. **Testing**: Ensure all tests pass with `python system_test.py`
4. **Documentation**: Update relevant documentation files
5. **Backwards Compatibility**: Maintain compatibility when possible

### Commit Message Format
```
type(scope): description

Examples:
feat(camera): add support for RTSP streams
fix(tracking): resolve duplicate counting issue
docs(readme): update installation instructions
test(system): add camera connectivity tests
```

### Release Process
1. Update VERSION file
2. Update this CHANGELOG.md
3. Run full test suite
4. Create git tag with version number
5. Update documentation if needed

---

**Note**: This project follows semantic versioning. Breaking changes will increment the major version, new features increment minor version, and bug fixes increment patch version.
