﻿#pragma checksum "..\..\..\..\MainWindow.xaml" "{ff1816ec-aa5e-4d10-87f7-6f4963833460}" "6690AA5327C4ECE7C98070D8158B5BAE0D67FE83"
//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated by a tool.
//     Runtime Version:4.0.30319.42000
//
//     Changes to this file may cause incorrect behavior and will be lost if
//     the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

using System;
using System.Diagnostics;
using System.Windows;
using System.Windows.Automation;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Controls.Ribbon;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Ink;
using System.Windows.Input;
using System.Windows.Markup;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Effects;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using System.Windows.Media.TextFormatting;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Shell;
using lecture_7_Neural_Networks_WPF;


namespace lecture_7_Neural_Networks_WPF {
    
    
    /// <summary>
    /// MainWindow
    /// </summary>
    public partial class MainWindow : System.Windows.Window, System.Windows.Markup.IComponentConnector {
        
        
        #line 10 "..\..\..\..\MainWindow.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Button btnBuildModel;
        
        #line default
        #line hidden
        
        
        #line 11 "..\..\..\..\MainWindow.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Label lblStatus1;
        
        #line default
        #line hidden
        
        
        #line 12 "..\..\..\..\MainWindow.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Button btnSplitDataTestTrain;
        
        #line default
        #line hidden
        
        
        #line 13 "..\..\..\..\MainWindow.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Button btnTestUseenData;
        
        #line default
        #line hidden
        
        
        #line 14 "..\..\..\..\MainWindow.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ListBox lstResults;
        
        #line default
        #line hidden
        
        
        #line 15 "..\..\..\..\MainWindow.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Button btnBuildNFoldCross;
        
        #line default
        #line hidden
        
        private bool _contentLoaded;
        
        /// <summary>
        /// InitializeComponent
        /// </summary>
        [System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [System.CodeDom.Compiler.GeneratedCodeAttribute("PresentationBuildTasks", "4.8.1.0")]
        public void InitializeComponent() {
            if (_contentLoaded) {
                return;
            }
            _contentLoaded = true;
            System.Uri resourceLocater = new System.Uri("/lecture 7 Neural Networks WPF;component/mainwindow.xaml", System.UriKind.Relative);
            
            #line 1 "..\..\..\..\MainWindow.xaml"
            System.Windows.Application.LoadComponent(this, resourceLocater);
            
            #line default
            #line hidden
        }
        
        [System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [System.CodeDom.Compiler.GeneratedCodeAttribute("PresentationBuildTasks", "4.8.1.0")]
        [System.ComponentModel.EditorBrowsableAttribute(System.ComponentModel.EditorBrowsableState.Never)]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Design", "CA1033:InterfaceMethodsShouldBeCallableByChildTypes")]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Maintainability", "CA1502:AvoidExcessiveComplexity")]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1800:DoNotCastUnnecessarily")]
        void System.Windows.Markup.IComponentConnector.Connect(int connectionId, object target) {
            switch (connectionId)
            {
            case 1:
            this.btnBuildModel = ((System.Windows.Controls.Button)(target));
            
            #line 10 "..\..\..\..\MainWindow.xaml"
            this.btnBuildModel.Click += new System.Windows.RoutedEventHandler(this.btnBuildModel_Click);
            
            #line default
            #line hidden
            return;
            case 2:
            this.lblStatus1 = ((System.Windows.Controls.Label)(target));
            return;
            case 3:
            this.btnSplitDataTestTrain = ((System.Windows.Controls.Button)(target));
            
            #line 12 "..\..\..\..\MainWindow.xaml"
            this.btnSplitDataTestTrain.Click += new System.Windows.RoutedEventHandler(this.btnSplitDataTestTrain_Click);
            
            #line default
            #line hidden
            return;
            case 4:
            this.btnTestUseenData = ((System.Windows.Controls.Button)(target));
            
            #line 13 "..\..\..\..\MainWindow.xaml"
            this.btnTestUseenData.Click += new System.Windows.RoutedEventHandler(this.btnTestUseenData_Click);
            
            #line default
            #line hidden
            return;
            case 5:
            this.lstResults = ((System.Windows.Controls.ListBox)(target));
            return;
            case 6:
            this.btnBuildNFoldCross = ((System.Windows.Controls.Button)(target));
            
            #line 15 "..\..\..\..\MainWindow.xaml"
            this.btnBuildNFoldCross.Click += new System.Windows.RoutedEventHandler(this.btnBuildModelNFoldCrossValidation);
            
            #line default
            #line hidden
            return;
            }
            this._contentLoaded = true;
        }
    }
}
