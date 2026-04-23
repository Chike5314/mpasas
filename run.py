"""
MPASAS – Entry Point
Run: python run.py
Then open: http://localhost:5000
"""
from app import create_app

if __name__ == '__main__':
    application = create_app()
    print("\n" + "═" * 55)
    print("  🎓  MPASAS  ·  MCQ Auto-Scoring & Analytics System")
    print("═" * 55)
    print("  Running at:  http://localhost:5000")
    print("  Stop with:   Ctrl + C")
    print("═" * 55 + "\n")
    application.run(debug=True, host='0.0.0.0', port=5000)
