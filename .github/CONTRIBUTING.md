# Contributing to Semantic Segmentation for Autonomous Navigation

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🎯 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected vs actual behavior**
- **Screenshots** if applicable
- **Environment details** (OS, browser, Python version, etc.)

**Example:**
```markdown
**Bug**: WebSocket connection fails on Firefox

**Steps to Reproduce:**
1. Open application in Firefox
2. Click START button
3. Observe console error

**Expected:** WebSocket connects successfully
**Actual:** Connection refused error

**Environment:** Firefox 120, macOS Sonoma
```

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Use case** - Why is this needed?
- **Proposed solution** - How should it work?
- **Alternatives considered** - What other approaches exist?

### Pull Requests

1. Fork the repository
2. Create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Test thoroughly
5. Commit with clear messages:
   ```bash
   git commit -m "feat: add real-time FPS counter"
   ```
6. Push and create a Pull Request

## 📋 Coding Guidelines

### Code Style

- **TypeScript/JavaScript**: Follow existing conventions
- **Python**: PEP 8 style guide
- **Components**: Use functional components with hooks
- **Naming**: Descriptive names for variables and functions

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Test additions/changes
- `chore:` - Build process or auxiliary tool changes

### Testing

Before submitting a PR:

```bash
# Run linter
bun run lint

# Build the project
bun run build

# Test the segmentation service
cd mini-services/segmentation-service
python index.py
```

## 🏗️ Architecture Overview

### Frontend Stack
- **Next.js 16** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **shadcn/ui** - UI components

### Backend Stack
- **FastAPI** - Python web framework
- **YOLOv8-Seg** - Instance segmentation model
- **WebSocket** - Real-time communication

### Project Structure
```
src/
├── app/           # Next.js pages and API routes
├── components/    # React components
├── hooks/         # Custom hooks
└── lib/           # Utilities and helpers

mini-services/
└── segmentation-service/
    └── index.py   # FastAPI backend
```

## 🚀 Development Setup

1. **Clone and install:**
   ```bash
   git clone https://github.com/yourusername/semantic-segmentation-navigation.git
   cd semantic-segmentation-navigation
   bun install
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Install Python dependencies:**
   ```bash
   cd mini-services/segmentation-service
   pip install -r requirements.txt
   ```

4. **Run development servers:**
   ```bash
   # Terminal 1 - Backend
   cd mini-services/segmentation-service
   python index.py

   # Terminal 2 - Frontend
   bun run dev
   ```

## 📝 Code Review Process

All PRs require:

1. ✅ Passing CI checks (lint, build)
2. ✅ At least one approval from a maintainer
3. ✅ No unresolved comments
4. ✅ Clear description of changes

## 🎨 UI/UX Guidelines

When contributing UI changes:

- Maintain the **retro terminal aesthetic**
- Ensure **accessibility** (color contrast, keyboard navigation)
- Test on **mobile and desktop**
- Keep the **performance** impact minimal

## 🔒 Security

- Never commit `.env` files or secrets
- Report security vulnerabilities privately
- Sanitize all user inputs
- Validate WebSocket messages

## 📞 Questions?

Open an issue for any questions or discussions.

---

Thank you for contributing to making autonomous navigation more accessible! 🚀
