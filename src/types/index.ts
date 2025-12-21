export type ModelType = 'yolo' | 'unet';

export interface AvailableModels {
  yolo?: boolean;
  unet?: boolean;
  default?: ModelType;
}

export interface Finding {
  id: string;
  label: string;
  confidence: number;
  segmentation: [number, number][];
}

export interface ChatMessage {
  id: string;
  sender: 'bot' | 'user';
  text: string;
  thinking?: boolean;
}
