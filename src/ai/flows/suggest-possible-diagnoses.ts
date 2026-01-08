'use server';

/**
 * @fileOverview This file defines a Genkit flow for suggesting possible diagnoses based on X-ray image analysis.
 *
 * - suggestPossibleDiagnoses - A function that suggests potential diagnoses based on X-ray image and analysis.
 * - SuggestPossibleDiagnosesInput - The input type for the suggestPossibleDiagnoses function.
 * - SuggestPossibleDiagnosesOutput - The return type for the suggestPossibleDiagnoses function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const SuggestPossibleDiagnosesInputSchema = z.object({
  xrayAnalysis: z
    .string()
    .describe(
      'The analysis of the X-ray, including findings and observations.'
    ),
  patientHistory: z
    .string()
    .optional()
    .describe('The patient medical history, if available.'),
});
export type SuggestPossibleDiagnosesInput = z.infer<
  typeof SuggestPossibleDiagnosesInputSchema
>;

const SuggestPossibleDiagnosesOutputSchema = z.object({
  possibleDiagnoses: z
    .array(z.string())
    .describe('A list of possible diagnoses based on the X-ray analysis.'),
  reasoning: z
    .string()
    .describe('The reasoning behind the suggested diagnoses.'),
});
export type SuggestPossibleDiagnosesOutput = z.infer<
  typeof SuggestPossibleDiagnosesOutputSchema
>;

export async function suggestPossibleDiagnoses(
  input: SuggestPossibleDiagnosesInput
): Promise<SuggestPossibleDiagnosesOutput> {
  return suggestPossibleDiagnosesFlow(input);
}

const prompt = ai.definePrompt({
  name: 'suggestPossibleDiagnosesPrompt',
  input: {schema: SuggestPossibleDiagnosesInputSchema},
  output: {schema: SuggestPossibleDiagnosesOutputSchema},
  prompt: `You are an expert dentist. Based on the X-ray analysis and patient history, suggest a list of possible diagnoses.

X-ray Analysis: {{{xrayAnalysis}}}
Patient History: {{{patientHistory}}}

Provide the list of possible diagnoses and the reasoning behind them. Be brief and to the point. Do not number the list.

{{output descriptions}}`,
});

const suggestPossibleDiagnosesFlow = ai.defineFlow(
  {
    name: 'suggestPossibleDiagnosesFlow',
    inputSchema: SuggestPossibleDiagnosesInputSchema,
    outputSchema: SuggestPossibleDiagnosesOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
