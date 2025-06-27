# Contributing to People Counting System

Thank you for your interest in contributing! This guide will help you get started with contributing to the People Counting System.

## 🎯 Project Vision

We aim to create a **reusable, well-documented, and future-proof** people counting system that:
- Works across different platforms and cameras
- Is easy to set up and use
- Provides accurate and reliable counting
- Can be extended with additional features
- Maintains user privacy and data security

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git
- Basic understanding of computer vision and/or web development

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/People-Counting-Yolo-MyVersion.git
   cd People-Counting-Yolo-MyVersion
   ```

2. **Set Up Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate (Windows)
   venv\Scripts\activate
   
   # Activate (macOS/Linux)
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Run Tests**
   ```bash
   python system_test.py  # Full system test
   python quick_test.py   # Quick health check
   ```

4. **Launch Application**
   ```bash
   streamlit run app.py
   ```

## 📋 Development Guidelines

### Code Style
- **PEP 8** compliance for Python code
- **Clear variable names** and function documentation
- **Type hints** where appropriate
- **Comprehensive comments** for complex logic

### Project Structure
```
People-Counting-Yolo-MyVersion/
├── 📱 app.py                      # Main Streamlit application
├── 🎥 camera.py                   # Live camera functionality  
├── 👤 face_recognition_system.py  # Face recognition module
├── 🎯 integrated_tracking.py      # Combined YOLO + Face recognition
├── 🧪 system_test.py              # Comprehensive testing
├── ⚡ quick_test.py               # Quick health checks
├── 📋 requirements.txt            # Project dependencies
├── 🌍 .env.example               # Environment configuration template
├── 📖 Documentation files        # README.md, guides, etc.
├── 🤖 models/                     # AI model logic
│   └── model.py                   # YOLO processing
├── 📁 data/                       # Input data and face database
├── 📁 results/                    # Output files
├── 🔧 scripts/                    # Utility scripts
├── 🧪 tests/                      # Test files
└── 📄 docs/                       # Additional documentation
```

### Coding Standards

#### Python Code
```python
def function_name(param: type) -> return_type:
    """
    Clear function description.
    
    Args:
        param: Description of parameter
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When this exception is raised
    """
    # Implementation with clear comments
    pass
```

#### Error Handling
```python
try:
    # Risky operation
    result = some_operation()
except SpecificException as e:
    logger.error(f"Specific error occurred: {e}")
    # Graceful fallback
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Generic fallback
```

#### Configuration
- Use `.env` files for configuration
- Provide `.env.example` with documentation
- Support environment variable overrides

## 🧪 Testing Requirements

### Before Submitting
All contributions must pass these tests:

1. **System Test**
   ```bash
   python system_test.py
   ```

2. **Quick Test**
   ```bash
   python quick_test.py
   ```

3. **Manual Testing**
   - Launch app successfully
   - Test with sample video
   - Test live camera (if available)
   - Verify face recognition (if dependencies installed)

### Test Coverage
- **Core Features**: YOLO detection, unique counting, camera handling
- **Optional Features**: Face recognition with graceful fallback
- **Error Handling**: Network failures, camera disconnection, missing dependencies
- **Cross-platform**: Windows, macOS, Linux compatibility

## 📝 Contribution Types

### 🐛 Bug Fixes
- Fix existing functionality
- Improve error handling
- Resolve compatibility issues

### ✨ New Features
- Additional camera types (IP, RTSP, etc.)
- New tracking algorithms
- Enhanced UI components
- Export/reporting capabilities
- Performance optimizations

### 📚 Documentation
- Code documentation
- User guides
- Setup instructions
- API documentation

### 🧪 Testing
- Unit tests
- Integration tests
- Performance benchmarks
- Cross-platform testing

## 🔄 Contribution Process

### 1. Issue Creation
Before coding, create an issue to discuss:
- **Bug reports**: Include steps to reproduce, expected vs actual behavior
- **Feature requests**: Include use case, proposed solution, alternatives considered
- **Questions**: Include context and what you've already tried

### 2. Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make Changes**
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation
   - Test thoroughly

3. **Commit Messages**
   ```
   type(scope): description
   
   Examples:
   feat(camera): add RTSP stream support
   fix(tracking): resolve duplicate counting in crowded scenes  
   docs(readme): update installation instructions for macOS
   test(system): add camera connectivity validation
   refactor(model): optimize YOLO inference performance
   ```

4. **Update Documentation**
   - Update relevant README sections
   - Add changelog entry
   - Update version if appropriate

### 3. Pull Request

1. **Before Submitting**
   - Ensure all tests pass
   - Update CHANGELOG.md
   - Rebase on latest main branch

2. **PR Description Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature  
   - [ ] Documentation update
   - [ ] Performance improvement
   - [ ] Other (specify)
   
   ## Testing
   - [ ] System tests pass
   - [ ] Manual testing completed
   - [ ] Cross-platform testing (if applicable)
   
   ## Checklist
   - [ ] Code follows project style guidelines
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated
   - [ ] Tests pass
   ```

3. **Review Process**
   - Maintain discussion in PR comments
   - Address reviewer feedback promptly
   - Keep PR focused and atomic

## 🏗️ Architecture Guidelines

### Modular Design
- **Separation of Concerns**: Each module has a single responsibility
- **Loose Coupling**: Modules interact through well-defined interfaces
- **High Cohesion**: Related functionality grouped together

### Configuration Management
- Environment-based configuration via `.env` files
- Runtime configuration through Streamlit UI
- Sensible defaults for all settings

### Error Handling
- **Graceful Degradation**: Optional features fail gracefully
- **User-Friendly Messages**: Clear, actionable error messages
- **Logging**: Appropriate logging levels for debugging

### Performance Considerations
- **Lazy Loading**: Load modules only when needed
- **Resource Management**: Proper cleanup of cameras and memory
- **Caching**: Cache expensive operations where appropriate

## 🔒 Security Guidelines

### Privacy First
- **Local Processing**: No data sent to external servers without explicit consent
- **Secure Storage**: Encrypt sensitive data (face encodings, personal info)
- **Access Control**: Implement proper permissions for sensitive features

### Data Handling
- **Minimal Collection**: Only collect necessary data
- **Clear Retention**: Define data retention policies
- **User Control**: Allow users to delete their data

## 📋 Release Process

### Version Numbering
Following [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

### Release Checklist
1. [ ] All tests pass
2. [ ] Documentation updated
3. [ ] CHANGELOG.md updated
4. [ ] VERSION file updated
5. [ ] Performance tested
6. [ ] Cross-platform tested
7. [ ] Security reviewed

## 💬 Communication

### Channels
- **GitHub Issues**: Bug reports, feature requests, questions
- **Pull Requests**: Code review and discussion
- **Discussions**: General questions and ideas

### Response Times
- **Issues**: We aim to respond within 48 hours
- **Pull Requests**: Initial review within 72 hours
- **Security Issues**: Within 24 hours

## 🎓 Learning Resources

### Computer Vision
- [OpenCV Documentation](https://docs.opencv.org/)
- [YOLO Papers and Implementations](https://github.com/ultralytics/ultralytics)

### Web Development
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Python Web Development Best Practices](https://realpython.com/python-web-applications/)

### Testing
- [Python Testing Best Practices](https://realpython.com/python-testing/)
- [Streamlit Testing](https://docs.streamlit.io/library/advanced-features/app-testing)

## 🙏 Recognition

Contributors will be:
- Listed in README.md acknowledgments
- Mentioned in release notes for significant contributions
- Invited to become maintainers for sustained contributions

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

**Questions?** Feel free to open an issue or start a discussion. We're here to help! 🚀
