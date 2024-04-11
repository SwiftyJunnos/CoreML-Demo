//
//  CoreML_DemoApp.swift
//  CoreML_Demo
//
//  Created by 이창준 on 4/4/24.
//

import SwiftUI

@main
struct CoreML_DemoApp: App {
    
    var body: some Scene {
        WindowGroup {
            let classifierModel = try! ViTMLPackage()
            let classifier = ClassifierService(classifierModel: classifierModel)
            ClassifierView(classifier: classifier)
        }
    }
    
}
