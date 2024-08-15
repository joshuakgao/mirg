import sys
import os
import pandas as pd
from pandas import DataFrame
import pprint as pp

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)  # for importing paths
from paths import ROOT_DIR


class Nbi:
    def __init__(
        self, path=ROOT_DIR + "/data/national_bridge_inventory/data/nbi.csv"
    ) -> dict:
        self.path = path
        self.data = self._preprocess(path)

    def _preprocess(self, path) -> DataFrame:
        renames = {
            # "Identification" section
            "STATE_CODE_001": "1_StateNames",
            "STRUCTURE_NUMBER_008": "8_StructureNumber",
            "RECORD_TYPE_005A": "5_InventoryRoute",
            "HIGHWAY_DISTRICT_002": "2_HighwayDistrict",
            "COUNTY_CODE_003": "3_CountyCode",
            "PLACE_CODE_004": "4_PlaceCode",
            "FEATURES_DESC_006A": "6_FeaturesIntersected",
            "FACILITY_CARRIED_007": "7_FacilityCarried",
            "LOCATION_009": "9_Location",
            "KILOPOINT_011": "11_MilePoint",
            "BASE_HWY_NETWORK_012": "12_BaseHighwayNetwork",
            "LRS_INV_ROUTE_013A": "13_LrsInventoryRte&Subrte",
            "LATDD": "16_Latitude",
            "LONGDD": "17_Longitude",
            "OTHER_STATE_CODE_098A": "98_BorderBridgeStateCode",
            "OTHR_STATE_STRUC_NO_099": "99_BorderBridgeStuctureNo",
            # "Structure Type and Material" section
            "STRUCTURE_KIND_043A": "43A_MainStructureMaterial",
            "STRUCTURE_TYPE_043B": "43B_MainStructureType",
            "APPR_KIND_044A": "44A_ApproachStructureMaterial",
            "APPR_TYPE_044B": "44B_ApproachStructureType",
            "MAIN_UNIT_SPANS_045": "45_NoOfSpansInMainUnit",
            "APPR_SPANS_046": "46_NoOfApproachSpans",
            "DECK_STRUCTURE_TYPE_107": "107_DeckStructureType",
            "SURFACE_TYPE_108A": "108A_WearingSurfaceProtectiveSystemTypeOfWearingSurface",
            "MEMBRANE_TYPE_108B": "108B_WearingSurfaceProtectiveSystemTypeOfMembrane",
            "DECK_PROTECTION_108C": "108C_WearingSurfaceProtectiveSystemTypeofDeckProtection",
            # "Age and Service" section
            "YEAR_BUILT_027": "27_YearBuilt",
            "YEAR_RECONSTRUCTED_106": "106_YearReconstructed",
            "SERVICE_ON_042A": "42A_TypeOfServiceOn",
            "SERVICE_UND_042B": "42B_TypeOfServiceUnder",
            "TRAFFIC_LANES_ON_028A": "28A_LaneOn",
            "TRAFFIC_LANES_UND_028B": "28B_LaneUnder",
            "ADT_029": "29_AverageDailyTraffic",
            "YEAR_ADT_030": "30_YearOfAdt",
            "PERCENT_ADT_TRUCK_109": "109_TruckAdt",
            "DETOUR_KILOS_019": "19_BypassDetourLength",
            # "Geometric Data"
            "MAX_SPAN_LEN_MT_048": "48_LengthOfMaximumSpan",
            "STRUCTURE_LEN_MT_049": "49_StructureLength",
            "LEFT_CURB_MT_050A": "50A_CurbOrSidewalkWidthLeft",
            "RIGHT_CURB_MT_050B": "50B_CurbOrSidewalkWidthRight",
            "ROADWAY_WIDTH_MT_051": "51_BridgeRoadwayWidthCurbToCurb",
            "DECK_WIDTH_MT_052": "52_DeckWidthOutToOut",
            "APPR_WIDTH_MT_032": "32_ApproachRoadwayWidth",
            "MEDIAN_CODE_033": "33_BridgeMedian",
            "DEGREES_SKEW_034": "34_Skew",
            "STRUCTURE_FLARED_035": "35_StructureFlared",
            "MIN_VERT_CLR_010": "10_InventoryRouteMinVertClear",
            "HORR_CLR_MT_047": "47_InventoryRouteTotalHorizClear",
            "VERT_CLR_OVER_MT_053": "53_MinVertClearOverBridgeRdwy",
            "VERT_CLR_UND_054B": "54B_MinVertUnderclear",
            "VERT_CLR_UND_REF_054A": "54A_MinVertUnderclearRef",
            "LAT_UND_MT_055B": "55B_MinLatUnderclearRt",
            "LAT_UND_REF_055A": "55A_MinLatUnderclearRtRef",
            "LEFT_LAT_UND_MT_056": "56_MinLatUnderclearLt",
            # "Navigation Data" section
            "NAVIGATION_038": "38_NavigationControl",
            "PIER_PROTECTION_111": "111_PierProtection",
            "NAV_VERT_CLR_MT_039": "39_NavigationVerticalClearance",
            "MIN_NAV_CLR_MT_116": "116_VertLiftBridgeNavMinVertClear",
            "NAV_HORR_CLR_MT_040": "40_NavigationHorizontalClearance",
            # "Classification" section
            "BRIDGE_LEN_IND_112": "112_NbisBridgeLength",
            "HIGHWAY_SYSTEM_104": "104_HighwaySystem",
            "FUNCTIONAL_CLASS_026": "26_FunctionalClass",
            "STRAHNET_HIGHWAY_100": "100_DefenseHighway",
            "PARALLEL_STRUCTURE_101": "101_ParallelStructure",
            "TRAFFIC_DIRECTION_102": "102_DirectionOfTraffic",
            "TEMP_STRUCTURE_103": "103_TemporaryStructure",
            "FEDERAL_LANDS_105": "105_FederalLandsHighways",
            "NATIONAL_NETWORK_110": "110_DesignatedNationalNetwork",
            "TOLL_020": "20_Toll",
            "MAINTENANCE_021": "21_Maintain",
            "OWNER_022": "22_Owner",
            "HISTORY_037": "37_HistoricalSignificance",
            # "Condition" section
            "DECK_COND_058": "58_Deck",
            "SUPERSTRUCTURE_COND_059": "59_Superstructure",
            "SUBSTRUCTURE_COND_060": "60_Substructure",
            "CHANNEL_COND_061": "61_Channel&ChannelProtection",
            "CULVERT_COND_062": "62_Culverts",
            # "Load Rating and Posting" section
            "DESIGN_LOAD_031": "31_DesignLoad",
            "OPR_RATING_METH_063": "63_OperatingRatingMethod",
            "OPERATING_RATING_064": "64_OperatingRating",
            "INV_RATING_METH_065": "65_InventoryRatingMethod",
            "INVENTORY_RATING_066": "66_InventoryRating",
            "POSTING_EVAL_070": "70_BridgePosting",
            "OPEN_CLOSED_POSTED_041": "41_StructureOpenPostedClosed",
            # "Appraisal" section
            "STRUCTURAL_EVAL_067": "67_StructuralEvaluation",
            "DECK_GEOMETRY_EVAL_068": "68_DeckGeomtry",
            "UNDCLRENCE_EVAL_069": "69_ClearancesVerticalHorizontal",
            "WATERWAY_EVAL_071": "71_WaterwayAdequacy",
            "APPR_ROAD_EVAL_072": "72_ApproachRoadwayAlignment",
            "RAILINGS_036A": "36A_BridgeRailings",
            "TRANSITIONS_036B": "36B_Transitions",
            "APPR_RAIL_036C": "36C_ApproachGuardrail",
            "APPR_RAIL_END_036D": "36D_ApproachGuardrailEnds",
            "SCOUR_CRITICAL_113": "113_ScourCriticalBridges",
            # "Proposed Improvements" section
            "FUTURE_ADT_114": "114_FutureAdt",
            "YEAR_OF_FUTURE_ADT_115": "115_YearOfFutureAdt",
            # "Inspections" section
            "DATE_OF_INSPECT_090": "90_InspectionDate",
            "INSPECT_FREQ_MONTHS_091": "91_Frequency",
            "FRACTURE_092A": "92A_FractureCriticalDetail",
            "UNDWATER_LOOK_SEE_092B": "92B_UnderwaterInspection",
            "SPEC_INSPECT_092C": "92C_OtherSpecialInspection",
        }

        columns_to_drop = ["LAT_016", "LONG_017", "x", "y"]

        df = pd.read_csv(path)
        df = df.rename(columns=renames)
        df = df.drop(columns=columns_to_drop)
        return df

    def get_bridge_by_structure_number(self, structure_number):
        bridge_data = self.data.loc[self.data["8_StructureNumber"] == structure_number]

        # convert kilometers to miles
        bridge_data["11_MilePoint"] = round(bridge_data["11_MilePoint"] * 0.621371, 3)
        bridge_data["19_BypassDetourLength"] = round(
            bridge_data["19_BypassDetourLength"] * 0.621371, 3
        )

        # format data
        bridge_data = bridge_data.astype(str)
        bridge_data = bridge_data.to_dict(orient="records")[0]

        return bridge_data


if __name__ == "__main__":
    nbi = Nbi()
    pp.pprint(nbi.get_bridge_by_structure_number("00000000000S702"))
