//
//  CoreML_DemoTests.swift
//  CoreML_DemoTests
//
//  Created by 이창준 on 4/4/24.
//

import CoreML
import XCTest
@testable import CoreML_Demo

final class CoreML_DemoTests: XCTestCase {
    
    override func setUp() async throws {
        
    }

    func test_ImageNetClasses_JSON파일_로드() throws {
        let array = MLShapedArray<Float>(repeating: .zero, shape: [1, 1_000])
        guard let maxIndex = array.scalars.enumerated().max(by: { $0.element < $1.element })?.offset else {
            XCTFail("")
            return
        }
        
        measure {
            guard let jsonURL = Bundle.main.url(forResource: "ImageNet_Classes", withExtension: "json") else {
                return
            }
            do {
                let jsonData = try Data(contentsOf: jsonURL)
                let dictionary = try JSONSerialization.jsonObject(with: jsonData, options: []) as? [String: String]
                let values = dictionary?.sorted(by: { $0.key < $1.key }).map { $0.value } ?? []
                print(values[maxIndex])
            } catch {
                XCTFail("\(error)")
            }
        }
        
    }

}
