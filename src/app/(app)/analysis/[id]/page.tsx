import AnalysisClient from "./analysis-client";
import { PlaceHolderImages } from "@/lib/placeholder-images";
import type { AnalysisResult } from "@/lib/types";

export default function AnalysisPage({
  params,
  searchParams,
}: {
  params: { id: string };
  searchParams: { [key: string]: string | string[] | undefined };
}) {
  const { id } = params;
  const dataParam = searchParams.data;
  
  let analysisData: AnalysisResult | undefined;

  if (typeof dataParam === "string") {
    try {
      analysisData = JSON.parse(decodeURIComponent(dataParam));
    } catch (e) {
      console.error("Falha ao analisar os dados da análise da URL", e);
    }
  }

  // Fallback to placeholder if URL data is missing or invalid
  if (!analysisData) {
    const placeholderImage = PlaceHolderImages.find((p) => p.id === id);
    if (placeholderImage) {
      analysisData = {
        ...placeholderImage,
        url: placeholderImage.imageUrl,
        keyFindings: "Nenhum dado de análise encontrado. Exibindo informações de exemplo.",
        priority: "medium",
      };
    }
  }

  if (!analysisData) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p>Análise de Raio-X não encontrada.</p>
      </div>
    );
  }

  return <AnalysisClient analysisData={analysisData} />;
}
