//
//  ClassifierView.swift
//  CoreML_Demo
//
//  Created by 이창준 on 4/4/24.
//

import CoreML
import PhotosUI
import SwiftUI

struct ClassifierView: View {
    
    // MARK: - Constants
    
    private enum Metric {
        static let imageSize: CGFloat = 300.0
        static let shadowSize: CGFloat = 8.0
    }
    
    // MARK: - Properties
    private let selectionHaptic: UISelectionFeedbackGenerator = {
        let generator = UISelectionFeedbackGenerator()
        generator.prepare()
        return generator
    }()
    
    private let notificationHaptic: UINotificationFeedbackGenerator = {
        let generator = UINotificationFeedbackGenerator()
        generator.prepare()
        return generator
    }()
    
    private let classifier: ClassifierService
    
    @State var selectedModel: Model = .vit
    
    @State var selectedItem: PhotosPickerItem? = nil
    @State var selectedItemData: Data? = nil
    
    @State var resultProb: Double?
    @State var resultLabel: String = ""
    
    @State var isInferring: Bool = false
    
    // MARK: - Initializer
    
    init(classifier: ClassifierService) {
        self.classifier = classifier
    }
    
    // MARK: - View
    
    var body: some View {
        ZStack {
            VStack {
                Picker("Model", selection: self.$selectedModel) {
                    ForEach(Model.allCases, id: \.self) {
                        Text($0.rawValue)
                    }
                }
                .pickerStyle(.segmented)
                .padding()
                
                Spacer()
            }
            
            VStack(spacing: 32.0) {
                PhotosPicker(selection: self.$selectedItem,
                             matching: .images,
                             preferredItemEncoding: .compatible,
                             photoLibrary: .shared()
                ) {
                    if let data = self.selectedItemData,
                       let image = UIImage(data: data) {
                        SelectedImage(uiimage: image)
                    } else {
                        Placeholder()
                    }
                }
                .photosPickerAccessoryVisibility(.hidden)
                
                Button(action: {
                    self.selectionHaptic.selectionChanged()
                    self.isInferring = true
                    
                    Task(priority: .userInitiated) {
                        guard let data = self.selectedItemData,
                              let image = UIImage(data: data)?.cgImage else {
                            return
                        }
                        
                        let (prob, label): (Double, String) = try await {
                            switch self.selectedModel {
                            case .cifar:
                                return try await self.classifyWithCifar(image: image)
                            case .vit:
                                return try await self.classifier.classify(image: image)
                            }
                        }()
                        self.resultProb = prob
                        self.resultLabel = label
                        
                        self.notificationHaptic.notificationOccurred(.success)
                        self.isInferring = false
                    }
                }, label: {
                    ButtonLabel(title: "뭐게요?")
                })
            }
            .onChange(of: self.selectedItem) { oldValue, newValue in
                Task {
                    self.resultProb = nil
                    self.resultLabel = ""
                    if let newData = try await newValue?.loadTransferable(type: Data.self) {
                        self.selectedItemData = newData
                    }
                }
            }
            
            if self.isInferring {
                VStack {
                    Spacer()
                    ProgressView()
                        .padding()
                }
            } else {
                OutputView(prob: self.resultProb,
                           resultLabel: self.resultLabel)
            }
        }
        .padding()
    }
    
    @ViewBuilder
    func Placeholder() -> some View {
        Image(systemName: "photo.badge.plus.fill")
            .resizable()
            .aspectRatio(contentMode: .fit)
            .padding(Metric.imageSize / 2.5)
            .foregroundStyle(.white)
            .frame(width: Metric.imageSize, height: Metric.imageSize)
            .background(.gray.secondary, in: RoundedRectangle(cornerRadius: 12.0))
            .shadow(radius: Metric.shadowSize, x: .zero, y: 4)
    }
    
    @ViewBuilder
    func SelectedImage(uiimage image: UIImage) -> some View {
        Image(uiImage: image)
            .resizable()
            .aspectRatio(contentMode: .fill)
            .frame(width: Metric.imageSize, height: Metric.imageSize)
            .clipShape(RoundedRectangle(cornerRadius: 12.0))
            .shadow(radius: Metric.shadowSize, x: .zero, y: 4)
    }
    
    @ViewBuilder
    func ButtonLabel(title: String) -> some View {
        Text(title)
            .font(.system(size: 17.0))
            .fontWeight(.bold)
            .foregroundStyle(.white)
            .padding(.horizontal, 64.0)
            .padding(.vertical, 15.0)
            .background(in: RoundedRectangle(cornerRadius: 12.0))
            .backgroundStyle(.tint)
    }
    
    @ViewBuilder
    func OutputView(prob: Double?, resultLabel: String) -> some View {
        VStack(spacing: 4.0) {
            Spacer()
            
            if let prob {
                Text("prob : \(prob) " + String(format: "(%.1f", prob * 100) + "%)")
                    .font(.system(size: 17.0))
                    .fontWeight(.regular)
            }
            
            if self.selectedModel == .cifar,
               let className = ClassLabel(rawValue: resultLabel) {
                Text(className.sound + " !")
            } else {
                Text(resultLabel)
            }
        }
        .font(.system(size: 24.0))
        .fontWeight(.bold)
    }
    
}

extension ClassifierView {
    
    private func classifyWithCifar(image: CGImage) async throws -> (Double, String) {
        let model = try CifarMLPackage()
        let input = try CifarMLPackageInput(imageWith: image)
        let output = try await model.prediction(input: input)
        let label = output.classLabel
        return (output.classLabel_probs[label]!, label)
    }
    
}

#Preview {
    let classifierModel = try! ViTMLPackage()
    let classifier = ClassifierService(classifierModel: classifierModel)
    return ClassifierView(classifier: classifier)
}
