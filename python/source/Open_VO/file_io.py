from typing import List
from .data_types import SurfaceData, SurfaceObjData, SurfaceImgData


class SurfaceDataWriterPy:
    def __init__(self, output_file: str):
        self.output_file = output_file

    def write_data(self, surface_data_list: List[SurfaceData]):
        try:
            with open(self.output_file, 'w', newline='') as file:
                # No explicit CSV writer if format is custom string concatenation
                for data in surface_data_list:
                    # Matches the C++ string concatenation format
                    line = (
                        f"{data.imgName},"
                        f"{data.imgW},{data.imgH},"
                        f"{data.imgTopLeftX},{data.imgTopLeftY},"
                        f"{data.imgBotRightX},{data.imgBotRightY},"
                        f"{data.objId},{data.objLabel},"
                        f"{data.bbX},{data.bbY},"
                        f"{data.bbW},{data.bbH},"
                        f"{data.objCoordX},{data.objCoordY}"
                        # Note: 'mappedTo' is not in the C++ writer's output string
                    )
                    file.write(line + '\n')
        except IOError as e:
            print(f"Error writing to file {self.output_file}: {e}")


class SurfaceDataReaderPy:
    def __init__(self, input_file: str, img_folder: str, output_file: str):
        # img_folder might not be used by the reader itself, but by classes using the data
        self.input_file = input_file
        self.img_folder = img_folder  # Store if needed later, not directly used by read methods
        self.processed_output_file = output_file  # File to read processed data from

    def _parse_line(self, line: str, num_expected_fields: int, data_type):
        parts = line.strip().split(',')
        if len(parts) < num_expected_fields:  # Be somewhat lenient if trailing commas exist
            # print(f"Warning: Line '{line}' has fewer fields than expected ({len(parts)} vs {num_expected_fields}). Skipping.")
            return None

        data_instance = data_type()
        try:
            if isinstance(data_instance, SurfaceData):
                data_instance.imgName = parts[0].strip()
                data_instance.imgW = int(parts[1])
                data_instance.imgH = int(parts[2])
                data_instance.imgTopLeftX = float(parts[3])
                data_instance.imgTopLeftY = float(parts[4])
                data_instance.imgBotRightX = float(parts[5])
                data_instance.imgBotRightY = float(parts[6])
                data_instance.objId = int(parts[7])
                data_instance.objLabel = int(parts[8])
                data_instance.bbX = int(parts[9])
                data_instance.bbY = int(parts[10])
                data_instance.bbW = int(parts[11])
                data_instance.bbH = int(parts[12])
                data_instance.objCoordX = float(parts[13])
                data_instance.objCoordY = float(parts[14])
                # 'mappedTo' is not read in C++ from this file
            elif isinstance(data_instance, SurfaceObjData):
                data_instance.imgName = parts[0].strip()
                data_instance.imgW = int(parts[1])
                data_instance.imgH = int(parts[2])
                data_instance.imgTopLeftX = float(parts[3])
                data_instance.imgTopLeftY = float(parts[4])
                data_instance.imgBotRightX = float(parts[5])
                data_instance.imgBotRightY = float(parts[6])
                data_instance.objLabel = int(parts[7])
                data_instance.bbX = int(parts[8])
                data_instance.bbY = int(parts[9])
                data_instance.bbW = int(parts[10])
                data_instance.bbH = int(parts[11])
            elif isinstance(data_instance, SurfaceImgData):
                data_instance.imgName = parts[0].strip()
                data_instance.imgW = int(parts[1])
                data_instance.imgH = int(parts[2])
                data_instance.imgTopLeftX = float(parts[3])
                data_instance.imgTopLeftY = float(parts[4])
                data_instance.imgBotRightX = float(parts[5])
                data_instance.imgBotRightY = float(parts[6])
            else:
                return None  # Should not happen
            return data_instance
        except (ValueError, IndexError) as e:
            # print(f"Error parsing line '{line}': {e}. Skipping.")
            return None

    def read_processed_data(self) -> List[SurfaceData]:
        surface_data_list: List[SurfaceData] = []
        try:
            with open(self.processed_output_file, 'r') as file:
                for line in file:
                    data = self._parse_line(line, 15, SurfaceData)
                    if data:
                        surface_data_list.append(data)
        except IOError as e:
            print(f"Error reading file {self.processed_output_file}: {e}")
        return surface_data_list

    def read_raw_labeled_data(self) -> List[SurfaceObjData]:
        surface_obj_data_list: List[SurfaceObjData] = []
        try:
            with open(self.input_file, 'r') as file:
                for line in file:
                    data = self._parse_line(line, 12, SurfaceObjData)
                    if data:
                        surface_obj_data_list.append(data)
        except IOError as e:
            print(f"Error reading file {self.input_file}: {e}")
        return surface_obj_data_list

    def read_raw_data(self) -> List[SurfaceImgData]:
        surface_img_data_list: List[SurfaceImgData] = []
        try:
            with open(self.input_file, 'r') as file:
                for line in file:
                    data = self._parse_line(line, 7, SurfaceImgData)
                    if data:
                        surface_img_data_list.append(data)
        except IOError as e:
            print(f"Error reading file {self.input_file}: {e}")
        return surface_img_data_list
