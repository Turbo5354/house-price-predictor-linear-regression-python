# 🏠 House Price Predictor

A simple linear regression model that predicts house prices based on various features like area, construction year, floor, rooms, and amenities.

## 📋 Overview

This project demonstrates a basic machine learning implementation using **Scikit-learn** to predict housing prices. The model is trained on a small dataset of houses with 8 features and uses **Linear Regression** to estimate property values.

## ✨ Features

The model considers the following features:
- **Area** (m²) - Size of the house
- **Construction Year** - Year the building was constructed
- **Floor** - Floor number of the unit
- **Rooms** - Number of rooms
- **Elevator** - Availability of elevator (0/1)
- **Parking** - Availability of parking (0/1)
- **Warehouse** - Availability of storage (0/1)
- **Renovation** - Renovation level (0 = none, 1 = partial, 2 = full)

## 🚀 Getting Started

### Prerequisites

- Python 3.6+
- Required packages:
  ```bash
  pip install pandas scikit-learn
