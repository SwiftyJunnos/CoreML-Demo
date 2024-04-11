//
//  ClassLabel.swift
//  CoreML_Demo
//
//  Created by 이창준 on 4/5/24.
//

import Foundation

enum ClassLabel: String {
    case plane
    case car
    case bird
    case cat
    case deer
    case dog
    case frog
    case horse
    case ship
    case truck
    
    var sound: String {
        switch self {
        case .plane: "✈️ 비행기"
        case .car: "🚘 자동차"
        case .bird: "🦜 새"
        case .cat: "😼 고양이"
        case .deer: "🦌 사슴"
        case .dog: "🐕 강아지"
        case .frog: "🐸 개구리"
        case .horse: "🐎 말"
        case .ship: "🚤 배"
        case .truck: "🚚 트럭"
        }
    }
}
