export const queue = (fn: () => any) => {
    setTimeout(fn);
};
