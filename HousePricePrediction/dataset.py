# -*- coding = utf-8 -*-
# @Time: 2025/10/21 20:16
# @Author: Zhihang Yi
# @File: dataset.py
# @Software: PyCharm

import torch
from torch.utils.data import Dataset
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class HousePriceDataset(Dataset):
    numerical_columns = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
        'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
        'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
        'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
        'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
    ]
    _mean = None
    _std = None

    def __init__(self, csv_path, split='train', mean=None, std=None):
        self.data = pd.read_csv(csv_path)
        self.split = split

        # 只对数值型特征做标准化
        num_df = self.data[self.numerical_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

        if split == 'train':
            # 训练集计算均值和标准差
            self.mean = num_df.mean()
            self.std = num_df.std().replace(0, 1)  # 防止除以0
            HousePriceDataset._mean = self.mean
            HousePriceDataset._std = self.std
        else:
            # 验证/测试集使用训练集的均值和标准差
            if mean is not None and std is not None:
                self.mean = mean
                self.std = std.replace(0, 1)
            elif HousePriceDataset._mean is not None and HousePriceDataset._std is not None:
                self.mean = HousePriceDataset._mean
                self.std = HousePriceDataset._std
            else:
                raise ValueError("Mean and std must be provided for non-training split.")

        # 标准化
        num_df = (num_df - self.mean) / self.std

        # 保存标准化后的数值特征
        for i, col in enumerate(self.numerical_columns):
            setattr(self, self._to_attr_name(col), num_df.iloc[:, i].values)

        # 2. categorical features
        self.zoning = self.data['MSZoning'].astype('category').cat.codes.values
        self.lot_shape = self.data['LotShape'].astype('category').cat.codes.values
        self.land_contour = self.data['LandContour'].astype('category').cat.codes.values
        self.utilities = self.data['Utilities'].astype('category').cat.codes.values
        self.lot_configuration = self.data['LotConfig'].astype('category').cat.codes.values
        self.land_slope = self.data['LandSlope'].astype('category').cat.codes.values
        self.neighborhood = self.data['Neighborhood'].astype('category').cat.codes.values
        self.condition1 = self.data['Condition1'].astype('category').cat.codes.values
        self.condition2 = self.data['Condition2'].astype('category').cat.codes.values
        self.dwelling_type = self.data['BldgType'].astype('category').cat.codes.values
        self.house_style = self.data['HouseStyle'].astype('category').cat.codes.values
        self.roof_style = self.data['RoofStyle'].astype('category').cat.codes.values
        self.roof_material = self.data['RoofMatl'].astype('category').cat.codes.values
        self.exterior_covering_first = self.data['Exterior1st'].astype('category').cat.codes.values
        self.exterior_covering_second = self.data['Exterior2nd'].astype('category').cat.codes.values
        self.masonry_veneer_type = self.data['MasVnrType'].astype('category').cat.codes.values
        self.exterior_quality = self.data['ExterQual'].astype('category').cat.codes.values
        self.exterior_condition = self.data['ExterCond'].astype('category').cat.codes.values
        self.foundation = self.data['Foundation'].astype('category').cat.codes.values
        self.basement_quality = self.data['BsmtQual'].astype('category').cat.codes.values
        self.basement_condition = self.data['BsmtCond'].astype('category').cat.codes.values
        self.basement_exposure = self.data['BsmtExposure'].astype('category').cat.codes.values
        self.basement_finish_type1 = self.data['BsmtFinType1'].astype('category').cat.codes.values
        self.basement_finish_type2 = self.data['BsmtFinType2'].astype('category').cat.codes.values
        self.heating = self.data['Heating'].astype('category').cat.codes.values
        self.heating_quality = self.data['HeatingQC'].astype('category').cat.codes.values
        self.central_air = self.data['CentralAir'].astype('category').cat.codes.values
        self.electrical = self.data['Electrical'].astype('category').cat.codes.values
        self.kitchen_quality = self.data['KitchenQual'].astype('category').cat.codes.values
        self.functionality = self.data['Functional'].astype('category').cat.codes.values
        self.fireplace_quality = self.data['FireplaceQu'].astype('category').cat.codes.values
        self.garage_type = self.data['GarageType'].astype('category').cat.codes.values
        self.garage_finished = self.data['GarageFinish'].astype('category').cat.codes.values
        self.garage_quality = self.data['GarageQual'].astype('category').cat.codes.values
        self.garage_condition = self.data['GarageCond'].astype('category').cat.codes.values
        self.paved_drive = self.data['PavedDrive'].astype('category').cat.codes.values
        self.sale_type = self.data['SaleType'].astype('category').cat.codes.values
        self.sale_condition = self.data['SaleCondition'].astype('category').cat.codes.values

        # label
        if self.split == 'train':
            self.sale_price = self.data['SalePrice'].values
        elif self.split == 'test':
            pass
        else:
            raise ValueError("split must be 'train' or 'test'")

    def _to_attr_name(self, col):
        # 将列名转为类属性名
        mapping = {
            'MSSubClass': 'sub_class',
            'LotFrontage': 'lot_frontage',
            'LotArea': 'lot_area',
            'OverallQual': 'overall_quality',
            'OverallCond': 'overall_condition',
            'YearBuilt': 'year_built',
            'YearRemodAdd': 'year_remodeled',
            'MasVnrArea': 'masonry_veneer_area',
            'BsmtFinSF1': 'basement_finished_area1',
            'BsmtFinSF2': 'basement_finished_area2',
            'BsmtUnfSF': 'unfinished_basement_area',
            'TotalBsmtSF': 'total_basement_area',
            '1stFlrSF': 'first_floor_area',
            '2ndFlrSF': 'second_floor_area',
            'LowQualFinSF': 'low_quality_finished_area',
            'GrLivArea': 'above_ground_living_area',
            'BsmtFullBath': 'basement_full_bathrooms',
            'BsmtHalfBath': 'basement_half_bathrooms',
            'FullBath': 'full_bathrooms',
            'HalfBath': 'half_bathrooms',
            'BedroomAbvGr': 'bedrooms_above_ground',
            'KitchenAbvGr': 'kitchens_above_ground',
            'TotRmsAbvGrd': 'total_rooms_above_ground',
            'Fireplaces': 'fireplaces',
            'GarageYrBlt': 'garage_built',
            'GarageCars': 'garage_cars',
            'GarageArea': 'garage_area',
            'WoodDeckSF': 'wood_deck_area',
            'OpenPorchSF': 'open_porch_area',
            'EnclosedPorch': 'enclosed_porch_area',
            '3SsnPorch': 'three_season_porch_area',
            'ScreenPorch': 'screen_porch_area',
            'PoolArea': 'pool_area',
            'MiscVal': 'miscellaneous_value',
            'MoSold': 'month_sold',
            'YrSold': 'year_sold'
        }
        return mapping[col]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # numerical features
        features = [
            torch.tensor(self.sub_class[idx], dtype=torch.float32),
            torch.tensor(self.lot_frontage[idx], dtype=torch.float32),
            torch.tensor(self.lot_area[idx], dtype=torch.float32),
            torch.tensor(self.overall_quality[idx], dtype=torch.float32),
            torch.tensor(self.overall_condition[idx], dtype=torch.float32),
            torch.tensor(self.year_built[idx], dtype=torch.float32),
            torch.tensor(self.year_remodeled[idx], dtype=torch.float32),
            torch.tensor(self.masonry_veneer_area[idx], dtype=torch.float32),
            torch.tensor(self.basement_finished_area1[idx], dtype=torch.float32),
            torch.tensor(self.basement_finished_area2[idx], dtype=torch.float32),
            torch.tensor(self.unfinished_basement_area[idx], dtype=torch.float32),
            torch.tensor(self.total_basement_area[idx], dtype=torch.float32),
            torch.tensor(self.first_floor_area[idx], dtype=torch.float32),
            torch.tensor(self.second_floor_area[idx], dtype=torch.float32),
            torch.tensor(self.low_quality_finished_area[idx], dtype=torch.float32),
            torch.tensor(self.above_ground_living_area[idx], dtype=torch.float32),
            torch.tensor(self.basement_full_bathrooms[idx], dtype=torch.float32),
            torch.tensor(self.basement_half_bathrooms[idx], dtype=torch.float32),
            torch.tensor(self.full_bathrooms[idx], dtype=torch.float32),
            torch.tensor(self.half_bathrooms[idx], dtype=torch.float32),
            torch.tensor(self.bedrooms_above_ground[idx], dtype=torch.float32),
            torch.tensor(self.kitchens_above_ground[idx], dtype=torch.float32),
            torch.tensor(self.total_rooms_above_ground[idx], dtype=torch.float32),
            torch.tensor(self.fireplaces[idx], dtype=torch.float32),
            torch.tensor(self.garage_built[idx], dtype=torch.float32),
            torch.tensor(self.garage_cars[idx], dtype=torch.float32),
            torch.tensor(self.garage_area[idx], dtype=torch.float32),
            torch.tensor(self.wood_deck_area[idx], dtype=torch.float32),
            torch.tensor(self.open_porch_area[idx], dtype=torch.float32),
            torch.tensor(self.enclosed_porch_area[idx], dtype=torch.float32),
            torch.tensor(self.three_season_porch_area[idx], dtype=torch.float32),
            torch.tensor(self.screen_porch_area[idx], dtype=torch.float32),
            torch.tensor(self.pool_area[idx], dtype=torch.float32),
            torch.tensor(self.miscellaneous_value[idx], dtype=torch.float32),
            torch.tensor(self.month_sold[idx], dtype=torch.float32),
            torch.tensor(self.year_sold[idx], dtype=torch.float32),
        ]
        # categorical features
        features += [
            torch.tensor(self.zoning[idx], dtype=torch.float32),
            torch.tensor(self.lot_shape[idx], dtype=torch.float32),
            torch.tensor(self.land_contour[idx], dtype=torch.float32),
            torch.tensor(self.utilities[idx], dtype=torch.float32),
            torch.tensor(self.lot_configuration[idx], dtype=torch.float32),
            torch.tensor(self.land_slope[idx], dtype=torch.float32),
            torch.tensor(self.neighborhood[idx], dtype=torch.float32),
            torch.tensor(self.condition1[idx], dtype=torch.float32),
            torch.tensor(self.condition2[idx], dtype=torch.float32),
            torch.tensor(self.dwelling_type[idx], dtype=torch.float32),
            torch.tensor(self.house_style[idx], dtype=torch.float32),
            torch.tensor(self.roof_style[idx], dtype=torch.float32),
            torch.tensor(self.roof_material[idx], dtype=torch.float32),
            torch.tensor(self.exterior_covering_first[idx], dtype=torch.float32),
            torch.tensor(self.exterior_covering_second[idx], dtype=torch.float32),
            torch.tensor(self.masonry_veneer_type[idx], dtype=torch.float32),
            torch.tensor(self.exterior_quality[idx], dtype=torch.float32),
            torch.tensor(self.exterior_condition[idx], dtype=torch.float32),
            torch.tensor(self.foundation[idx], dtype=torch.float32),
            torch.tensor(self.basement_quality[idx], dtype=torch.float32),
            torch.tensor(self.basement_condition[idx], dtype=torch.float32),
            torch.tensor(self.basement_exposure[idx], dtype=torch.float32),
            torch.tensor(self.basement_finish_type1[idx], dtype=torch.float32),
            torch.tensor(self.basement_finish_type2[idx], dtype=torch.float32),
            torch.tensor(self.heating[idx], dtype=torch.float32),
            torch.tensor(self.heating_quality[idx], dtype=torch.float32),
            torch.tensor(self.central_air[idx], dtype=torch.float32),
            torch.tensor(self.electrical[idx], dtype=torch.float32),
            torch.tensor(self.kitchen_quality[idx], dtype=torch.float32),
            torch.tensor(self.functionality[idx], dtype=torch.float32),
            torch.tensor(self.fireplace_quality[idx], dtype=torch.float32),
            torch.tensor(self.garage_type[idx], dtype=torch.float32),
            torch.tensor(self.garage_finished[idx], dtype=torch.float32),
            torch.tensor(self.garage_quality[idx], dtype=torch.float32),
            torch.tensor(self.garage_condition[idx], dtype=torch.float32),
            torch.tensor(self.paved_drive[idx], dtype=torch.float32),
            torch.tensor(self.sale_type[idx], dtype=torch.float32),
            torch.tensor(self.sale_condition[idx], dtype=torch.float32),
        ]

        features = torch.stack(features)

        if self.split == 'train':
            label = torch.tensor(self.sale_price[idx], dtype=torch.float32)

            if torch.isnan(features).any():
                logger.error(f'NaN detected in features at index {idx}.')

            if torch.isnan(label).any():
                logger.error(f'NaN detected in label at index {idx}.')

            return features, label  # (74,), (1,)
        elif self.split == 'test':
            if torch.isnan(features).any():
                logger.error(f'NaN detected in features at index {idx}.')

            return features  # (74,)
        else:
            raise ValueError("split must be 'train' or 'test'")
