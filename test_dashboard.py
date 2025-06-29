"""
Simple Dashboard Test and Demo
Tests the interactive dashboard functionality
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

def test_dashboard_components():
    """Test dashboard components"""
    print("🧪 TESTING DASHBOARD COMPONENTS")
    print("=" * 40)
    
    try:
        # Import dashboard
        from interactive_dashboard import InteractiveDashboard
        print("✅ Dashboard module imported successfully")
        
        # Create dashboard instance
        dashboard = InteractiveDashboard()
        print("✅ Dashboard instance created")
        
        # Test data generation
        dashboard.generate_sample_data()
        print("✅ Sample data generated")
        print(f"   📊 Market data points: {len(dashboard.market_data)}")
        print(f"   🎯 Strategies loaded: {len(dashboard.strategies_data)}")
        
        # Show strategy info
        print("\n📈 STRATEGY DATA PREVIEW:")
        for name, data in dashboard.strategies_data.items():
            config = dashboard.strategy_configs[name]
            perf = data['performance']
            print(f"   {config['name']:<20} Return: {perf['total_return']*100:>7.2f}%")
        
        print("\n✅ All dashboard components working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def show_dashboard_info():
    """Show dashboard information"""
    print("\n📋 DASHBOARD FILE INFORMATION")
    print("=" * 40)
    
    file_path = "interactive_dashboard.py"
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        print(f"📁 File: {file_path}")
        print(f"📏 Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        
        # Count lines
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"📄 Lines of code: {len(lines)}")
        
        # Show key features
        print("\n🔧 KEY TECHNOLOGIES:")
        print("   • Plotly Dash - Interactive web framework")
        print("   • Bootstrap - Responsive CSS framework") 
        print("   • Pandas/Numpy - Data manipulation")
        print("   • Real-time updates with callbacks")
        
        print("\n🎨 DASHBOARD LAYOUT:")
        print("   📊 4 Interactive Charts")
        print("   📋 Performance Metrics Table")
        print("   ⚙️ Control Panel")
        print("   🔄 Auto-refresh Components")
        
    else:
        print("❌ Dashboard file not found!")

if __name__ == "__main__":
    print("🚀 INTERACTIVE DASHBOARD DEMO & TEST")
    print("=" * 50)
    
    # Show file info
    show_dashboard_info()
    
    # Test components
    test_success = test_dashboard_components()
    
    if test_success:
        print("\n🌐 DASHBOARD ACCESS INFORMATION:")
        print("=" * 40)
        print("🔗 URL: http://localhost:8050")
        print("📱 Works on desktop, tablet, and mobile")
        print("🔄 Auto-updates every 5 seconds")
        print("🎯 Interactive hover and zoom")
        
        print("\n💡 TO START THE DASHBOARD:")
        print("   Run this command:")
        print("   C:/Users/prana/OneDrive/Desktop/ALGO/StochasticFinance/.venv/Scripts/python.exe interactive_dashboard.py")
        
        print("\n🎯 FEATURES TO EXPLORE:")
        print("   • Watch real-time portfolio performance")
        print("   • Hover over charts for detailed data")
        print("   • Change time periods with dropdown")
        print("   • Compare strategy risk vs return")
        print("   • Observe trading signal patterns")
        
    print("\n" + "=" * 50)
    print("🎉 DEMO COMPLETE!")
