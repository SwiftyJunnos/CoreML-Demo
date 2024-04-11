//
//  ClassifierService.swift
//  CoreML_Demo
//
//  Created by 이창준 on 4/9/24.
//

import CoreGraphics
import Foundation

enum ClassifierError: Error {
    case outputError(_ description: String = "")
    case classError(_ description: String = "")
    case fileNotFound(_ description: String = "")
}

actor ClassifierService {
    typealias ClassifyResult = (probability: Double, label: String)
    
    // MARK: - Constants
    
    private enum Typo {
        static let classesFileName = "ImageNet_Classes"
    }
    
    // MARK: - Properties
    
    let classifierModel: ViTMLPackage
    
    // MARK: - Initializer
    
    init(classifierModel: ViTMLPackage) {
        self.classifierModel = classifierModel
    }
    
    // MARK: - Functions
    
    func classify(image: consuming CGImage) async throws -> ClassifyResult {
        try Task.checkCancellation()
        
        let input = try ViTMLPackageInput(imageWith: image)
        let output = try await self.classifierModel.prediction(input: input)
        
        // Index로 매핑할 class 목록 fetch
        guard let jsonURL = Bundle.main.url(forResource: Typo.classesFileName, withExtension: "json") else {
            throw ClassifierError.fileNotFound("\(Typo.classesFileName).json 파일을 찾을 수 없습니다.")
        }
        let classes = try self.fetchClasses(from: jsonURL)
        
        // (1, 1000) 결과에서 가장 큰 값을 찾아 그 인덱스를 휙득
        let predictions = output.predictionShapedArray.scalars
        guard let maxIndex = predictions.enumerated().max(by: { $0.element < $1.element })?.offset else {
            throw ClassifierError.outputError("Predict 결과에서 가장 큰 값을 찾을 수 없습니다.")
        }
        
        let probability = Double(predictions[maxIndex]) // score
        let label = classes[maxIndex] // label
        
        return (probability: probability, label: label)
    }
    
    private func fetchClasses(from jsonURL: consuming URL) throws -> [String] {
        let jsonData = try Data(contentsOf: jsonURL)
        guard let dictionary = try JSONSerialization.jsonObject(with: jsonData) as? [String: String] else {
            throw ClassifierError.classError("JSON 오브젝트를 [String: String] 타입의 딕셔너리로 변환할 수 없습니다.")
        }
        
        let values = dictionary.sorted(by: { $0.key < $1.key }).map { $0.value }
        return values
    }
    
}
